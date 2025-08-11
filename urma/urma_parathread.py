import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pygrib
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
import logging
from contextlib import contextmanager
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings
from pathlib import Path
import calendar
import json
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import uuid
from queue import Queue
import multiprocessing as mp

# Configure logging for thread-safe operation
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(threadName)s - %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Thread-local storage for S3 clients
thread_local_data = threading.local()

def get_s3_client():
    """Get thread-local S3 client for concurrent operations."""
    if not hasattr(thread_local_data, 's3_client'):
        thread_local_data.s3_client = boto3.client(
            's3', 
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=50,
                retries={'max_attempts': 3}
            )
        )
    return thread_local_data.s3_client

def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable objects to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

class ThreadSafeProgressTracker:
    """Thread-safe progress tracking for parallel operations."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.description = description
        self.pbar = tqdm(total=total_items, desc=description, 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
    def update(self, success: bool = True, message: str = None):
        """Thread-safe update of progress."""
        with self.lock:
            if success:
                self.completed_items += 1
            else:
                self.failed_items += 1
            
            self.pbar.update(1)
            if message:
                self.pbar.set_postfix_str(message)
    
    def close(self):
        """Close progress bar and print summary."""
        self.pbar.close()
        elapsed = time.time() - self.start_time
        success_rate = (self.completed_items / self.total_items) * 100
        logger.info(f"{self.description} complete: {self.completed_items}/{self.total_items} "
                   f"({success_rate:.1f}%) in {elapsed:.1f}s")

class URMAMonthlyExtractorParallel:
    """
    Parallel-optimized version of URMA data extractor with pre-calculated station indices.
    Uses ThreadPoolExecutor for I/O operations and supports concurrent processing.
    """

    def __init__(self, output_dir: str = "urma_data", max_workers: int = None, 
                 max_month_workers: int = None):
        """
        Initialize the parallel URMA data extractor.
        
        Parameters:
        -----------
        output_dir : str
            Directory for output files
        max_workers : int
            Maximum workers for timestamp processing (default: min(32, cpu_count + 4))
        max_month_workers : int
            Maximum workers for month processing (default: min(4, cpu_count))
        """
        self.bucket_name = 'noaa-urma-pds'
        self._grid_cache = {}
        
        # NEW: Pre-calculated station indices for each grid configuration  
        self._station_indices_cache = {}
        self._grid_stations_initialized = {}  # Track which grid/station combos are initialized
        self._cache_lock = threading.Lock()

        # Setup parallelism
        cpu_count = mp.cpu_count()
        self.max_workers = max_workers or min(32, cpu_count + 4)
        self.max_month_workers = max_month_workers or min(4, cpu_count)

        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print(f"✓ Parallel URMA Extractor initialized with pre-calculated indices optimization")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"🔄 Max workers (timestamps): {self.max_workers}")
        print(f"🔄 Max workers (months): {self.max_month_workers}")

    @contextmanager
    def _fetch_grib_data(self, timestamp: datetime):
        """Thread-safe GRIB data fetching with unique temporary files."""
        temp_file_path = None
        grib_messages = None
        thread_id = threading.get_ident()

        try:
            # Generate S3 key
            date_str = timestamp.strftime('%Y%m%d')
            hour_str = timestamp.strftime('%H')
            s3_key = f"urma2p5.{date_str}/urma2p5.t{hour_str}z.2dvaranl_ndfd.grb2_wexp"

            # Get thread-local S3 client
            s3_client = get_s3_client()

            # Download data
            response = s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            grib_data = response['Body'].read()

            # Create unique temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f'_{thread_id}_{uuid.uuid4().hex[:8]}.grb2'
            )
            temp_file.write(grib_data)
            temp_file.close()
            temp_file_path = temp_file.name

            # Open with pygrib
            grib_messages = pygrib.open(temp_file_path)
            yield grib_messages

        except Exception as e:
            logger.error(f"Error fetching data for {timestamp} (thread {thread_id}): {e}")
            yield None
        finally:
            if grib_messages:
                try:
                    grib_messages.close()
                except:
                    pass
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

    def _get_grid_coordinates(self, grib_message):
        """Thread-safe grid coordinate caching."""
        cache_key = f"{grib_message.Ni}_{grib_message.Nj}_{grib_message.gridType}"

        # Check cache with lock
        with self._cache_lock:
            if cache_key in self._grid_cache:
                return self._grid_cache[cache_key]

        # Generate coordinates (only one thread will do this)
        logger.debug("📍 Generating grid coordinates (one-time setup)...")
        lats, lons = grib_message.latlons()
        
        # Cache with lock
        with self._cache_lock:
            if cache_key not in self._grid_cache:  # Double-check in case another thread added it
                self._grid_cache[cache_key] = (lats, lons)
                logger.info(f"✓ Grid cached: {lats.shape}")

        return self._grid_cache[cache_key]

    def _initialize_station_indices(self, station_lats, station_lons, grib_message):
        """
        NEW: Thread-safe initialization of station indices for all grids once at the beginning.
        This replaces the repeated calls to _find_station_indices in parallel threads.
        """
        grid_cache_key = f"{grib_message.Ni}_{grib_message.Nj}_{grib_message.gridType}"
        stations_hash = str(hash(f"{station_lats.tobytes()}{station_lons.tobytes()}"))
        
        # Create combined key for this specific combination of grid and stations
        combined_key = f"{grid_cache_key}_{stations_hash}"
        
        # Check if already initialized with thread safety
        with self._cache_lock:
            if combined_key in self._station_indices_cache:
                return self._station_indices_cache[combined_key]

        # Only one thread should perform the calculation
        logger.info(f"🎯 Initializing station indices for {len(station_lats)} stations (thread-safe, one-time setup)...")

        # Get grid coordinates
        grid_lats, grid_lons = self._get_grid_coordinates(grib_message)

        # Flatten grid and create spatial index
        grid_shape = grid_lats.shape
        grid_coords = np.column_stack((grid_lats.flatten(), grid_lons.flatten()))
        tree = cKDTree(grid_coords)

        # Find nearest neighbors
        station_coords = np.column_stack((station_lats, station_lons))
        distances, indices = tree.query(station_coords)

        # Convert to 2D indices
        i_indices = indices // grid_shape[1]
        j_indices = indices % grid_shape[1]
        grid_indices = np.column_stack((i_indices, j_indices))

        # Cache results with thread safety
        with self._cache_lock:
            if combined_key not in self._station_indices_cache:  # Double-check pattern
                self._station_indices_cache[combined_key] = grid_indices
                self._grid_stations_initialized[combined_key] = True
                logger.info(f"✓ Station indices initialized - Max distance to grid: {distances.max():.4f}°")

        return self._station_indices_cache[combined_key]

    def _get_cached_station_indices(self, station_lats, station_lons, grib_message):
        """
        NEW: Get pre-calculated station indices. If not available, initialize them.
        This method is thread-safe and optimized for parallel access.
        """
        grid_cache_key = f"{grib_message.Ni}_{grib_message.Nj}_{grib_message.gridType}"
        stations_hash = str(hash(f"{station_lats.tobytes()}{station_lons.tobytes()}"))
        combined_key = f"{grid_cache_key}_{stations_hash}"
        
        # Fast path: check if already cached
        with self._cache_lock:
            if combined_key in self._station_indices_cache:
                return self._station_indices_cache[combined_key]
        
        # Slow path: initialize if not cached
        return self._initialize_station_indices(station_lats, station_lons, grib_message)

    def _safe_extract_values(self, values, station_indices):
        """Safely extract station values, handling masked arrays and missing data."""
        station_values = []

        for i, j in station_indices:
            try:
                # Handle potential out-of-bounds indices
                if i >= values.shape[0] or j >= values.shape[1] or i < 0 or j < 0:
                    station_values.append(np.nan)
                    continue

                value = values[i, j]

                # Handle masked arrays (suppress the warning)
                if np.ma.is_masked(value):
                    station_values.append(np.nan)
                elif hasattr(value, 'mask') and value.mask:
                    station_values.append(np.nan)
                else:
                    # Convert to regular float to avoid any masked array issues
                    station_values.append(float(value))

            except (IndexError, ValueError, TypeError):
                # Any extraction error results in NaN
                station_values.append(np.nan)

        return np.array(station_values, dtype=np.float64)

    def _extract_variables(self, grib_messages, station_indices):
        """Extract all variables from GRIB file with improved error handling."""
        extracted_data = {}
        variable_errors = []

        for msg_num, msg in enumerate(grib_messages, 1):
            try:
                var_name = msg.name
                level = getattr(msg, 'level', 'surface')
                level_type = getattr(msg, 'typeOfLevel', 'unknown')
                units = getattr(msg, 'units', 'unknown')

                # Create variable name
                if level_type != 'unknown' and str(level) != 'surface':
                    full_var_name = f"{var_name}_{level}_{level_type}"
                else:
                    full_var_name = var_name

                # Get values and handle masked arrays safely
                values = msg.values

                # Use safe extraction method
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    station_values = self._safe_extract_values(values, station_indices)

                # Check if we got valid data
                valid_count = np.sum(~np.isnan(station_values))

                extracted_data[full_var_name] = {
                    'values': station_values,
                    'units': units,
                    'level': level,
                    'level_type': level_type,
                    'valid_count': int(valid_count),
                    'total_count': int(len(station_values))
                }

            except Exception as e:
                error_msg = f"Message {msg_num} ({getattr(msg, 'name', 'unknown')}): {e}"
                variable_errors.append(error_msg)
                continue

        if variable_errors and len(variable_errors) > 5:
            logger.warning(f"Skipped {len(variable_errors)} variables due to errors")

        return extracted_data

    def _process_single_timestamp(self, args):
        """
        Process a single timestamp - designed for parallel execution with optimized station indices.
        NEW: Uses pre-calculated station indices instead of recalculating each time.
        """
        timestamp, metadata_df, lat_col, lon_col, station_id_col = args
        
        try:
            # Extract station info
            station_lats = metadata_df[lat_col].values
            station_lons = metadata_df[lon_col].values
            station_ids = metadata_df[station_id_col].tolist()

            with self._fetch_grib_data(timestamp) as grib_messages:
                if grib_messages is None:
                    return None, timestamp, "Failed to fetch GRIB data"

                # NEW: Get pre-calculated station indices (thread-safe)
                first_msg = grib_messages.message(1)
                station_indices = self._get_cached_station_indices(
                    station_lats, station_lons, first_msg
                )

                # Extract variables using pre-calculated indices
                grib_messages.rewind()
                extracted_data = self._extract_variables(grib_messages, station_indices)

                if not extracted_data:
                    return None, timestamp, "No variables extracted"

                # Create DataFrame for this timestamp
                times = [timestamp] * len(station_ids)
                index = pd.MultiIndex.from_arrays(
                    [times, station_ids],
                    names=['valid_time', 'station_id']
                )

                df_data = {var: info['values'] for var, info in extracted_data.items()}
                df_single = pd.DataFrame(df_data, index=index)

                # Return DataFrame and metadata
                variable_metadata = {
                    var: {k: v for k, v in info.items() if k != 'values'}
                    for var, info in extracted_data.items()
                }

                return df_single, timestamp, variable_metadata

        except Exception as e:
            error_msg = f"Error processing {timestamp}: {str(e)}"
            logger.error(error_msg)
            return None, timestamp, error_msg

    def _group_timestamps_by_month(self, timestamps: List[datetime]) -> Dict[Tuple[int, int], List[datetime]]:
        """Group timestamps by (year, month)."""
        monthly_groups = {}

        for timestamp in timestamps:
            month_key = (timestamp.year, timestamp.month)
            if month_key not in monthly_groups:
                monthly_groups[month_key] = []
            monthly_groups[month_key].append(timestamp)

        # Sort timestamps within each month
        for month_key in monthly_groups:
            monthly_groups[month_key].sort()

        return monthly_groups

    def _generate_filename(self, year: int, month: int, timestamps: List[datetime]) -> str:
        """Generate filename based on year, month, and actual date range."""
        first_date = min(timestamps).strftime('%Y%m%d')
        last_date = max(timestamps).strftime('%Y%m%d')
        month_name = calendar.month_abbr[month].lower()

        return f"urma_{year}_{month:02d}_{month_name}_{first_date}_to_{last_date}.parquet"

    def _file_exists(self, filename: str) -> bool:
        """Check if output file already exists."""
        filepath = self.output_dir / filename
        return filepath.exists()

    def _save_monthly_data(self, df: pd.DataFrame, filename: str):
        """Save monthly DataFrame to parquet file with proper metadata handling."""
        filepath = self.output_dir / filename

        try:
            # Save the main dataframe
            df.to_parquet(filepath, compression='snappy')
            logger.info(f"💾 Saved: {filename} ({df.shape[0]} rows, {df.shape[1]} columns)")

            # Save metadata separately with proper serialization
            metadata_file = filepath.with_suffix('.metadata.json')

            # Convert all metadata to JSON-serializable format
            serializable_metadata = convert_to_serializable(df.attrs)

            with open(metadata_file, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)

            logger.debug(f"Saved metadata to {metadata_file.name}")

        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
            raise

    def _pre_initialize_station_indices(self, metadata_df: pd.DataFrame, timestamps: List[datetime],
                                       lat_col: str, lon_col: str, station_id_col: str):
        """
        NEW: Pre-initialize station indices by downloading the first available GRIB file.
        This ensures all subsequent parallel processing uses cached indices.
        """
        station_lats = metadata_df[lat_col].values
        station_lons = metadata_df[lon_col].values
        
        logger.info(f"🔧 Pre-initializing station indices for optimal parallel performance...")
        
        # Try to get a GRIB file to establish the grid
        for timestamp in timestamps[:5]:  # Try first 5 timestamps
            try:
                with self._fetch_grib_data(timestamp) as grib_messages:
                    if grib_messages is not None:
                        first_msg = grib_messages.message(1)
                        # This will initialize and cache the station indices
                        self._initialize_station_indices(station_lats, station_lons, first_msg)
                        logger.info(f"✅ Station indices pre-initialized using {timestamp}")
                        return True
            except Exception as e:
                logger.debug(f"Failed to pre-initialize with {timestamp}: {e}")
                continue
        
        logger.warning("Could not pre-initialize station indices - will initialize on first successful download")
        return False

    def _extract_month_data_parallel(self, metadata_df: pd.DataFrame, timestamps: List[datetime],
                                   lat_col: str, lon_col: str, station_id_col: str) -> Optional[pd.DataFrame]:
        """Extract data for a single month using parallel processing with optimized station indices."""
        
        start_time = time.time()
        month_name = calendar.month_name[timestamps[0].month]
        year = timestamps[0].year
        
        logger.info(f"🚀 Starting optimized parallel processing for {month_name} {year} "
                   f"({len(timestamps)} timestamps, {self.max_workers} workers)")

        # NEW: Pre-initialize station indices for this month
        self._pre_initialize_station_indices(metadata_df, timestamps, lat_col, lon_col, station_id_col)

        # Prepare arguments for parallel processing
        args_list = [(ts, metadata_df, lat_col, lon_col, station_id_col) for ts in timestamps]

        # Process timestamps in parallel
        all_dataframes = []
        failed_timestamps = []
        variable_metadata = None
        
        # Create progress tracker
        progress = ThreadSafeProgressTracker(
            len(timestamps), 
            f"{month_name} {year} timestamps"
        )

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_timestamp = {
                    executor.submit(self._process_single_timestamp, args): args[0] 
                    for args in args_list
                }

                # Collect results as they complete
                for future in as_completed(future_to_timestamp):
                    timestamp = future_to_timestamp[future]
                    
                    try:
                        result = future.result()
                        df_single, processed_timestamp, metadata_or_error = result

                        if df_single is not None:
                            all_dataframes.append(df_single)
                            
                            # Store metadata from first success
                            if variable_metadata is None and isinstance(metadata_or_error, dict):
                                variable_metadata = metadata_or_error
                            
                            progress.update(True, f"✓ {processed_timestamp.strftime('%m-%d %H:%M')}")
                        else:
                            failed_timestamps.append(processed_timestamp)
                            progress.update(False, f"✗ {processed_timestamp.strftime('%m-%d %H:%M')}")
                            logger.debug(f"Failed {processed_timestamp}: {metadata_or_error}")

                    except Exception as e:
                        failed_timestamps.append(timestamp)
                        progress.update(False, f"✗ {timestamp.strftime('%m-%d %H:%M')}")
                        logger.error(f"Future error for {timestamp}: {e}")

        finally:
            progress.close()

        if not all_dataframes:
            logger.error(f"No data extracted for {month_name} {year}")
            return None

        # Combine results for this month
        logger.info(f"📊 Combining {len(all_dataframes)} DataFrames...")
        month_df = pd.concat(all_dataframes, axis=0).sort_index()

        # Add metadata with proper type conversion
        month_df.attrs = {
            'variable_metadata': variable_metadata,
            'source': 'NOAA URMA',
            'extraction_info': {
                'total_timestamps': int(len(timestamps)),
                'successful_timestamps': int(len(all_dataframes)),
                'failed_timestamps': int(len(failed_timestamps)),
                'stations': int(len(metadata_df)),
                'month_start': min(timestamps).isoformat(),
                'month_end': max(timestamps).isoformat(),
                'processing_time_seconds': round(time.time() - start_time, 2),
                'max_workers': self.max_workers,
                'optimization_note': 'Station indices pre-calculated and cached for parallel processing'
            }
        }

        success_rate = (len(all_dataframes) / len(timestamps)) * 100
        elapsed = time.time() - start_time
        
        logger.info(f"✅ {month_name} {year} complete: {len(all_dataframes)}/{len(timestamps)} "
                   f"({success_rate:.1f}%) in {elapsed:.1f}s")

        return month_df

    def _process_single_month(self, args):
        """Process a single month - designed for parallel month processing."""
        month_key, month_timestamps, metadata_df, lat_col, lon_col, station_id_col, force_reprocess = args
        
        year, month = month_key
        month_name = calendar.month_name[month]
        
        try:
            # Generate filename
            filename = self._generate_filename(year, month, month_timestamps)

            # Check if file already exists
            if self._file_exists(filename) and not force_reprocess:
                logger.info(f"⏭️  Skipping {filename} (already exists)")
                return 'skipped', filename

            # Extract data for this month
            month_df = self._extract_month_data_parallel(
                metadata_df, month_timestamps, lat_col, lon_col, station_id_col
            )

            if month_df is not None:
                # Save the monthly data
                self._save_monthly_data(month_df, filename)
                return 'success', filename
            else:
                logger.error(f"❌ Failed to extract data for {month_name} {year}")
                return 'failed', filename

        except Exception as e:
            logger.error(f"Error processing {month_name} {year}: {e}")
            return 'error', filename

    def extract_data_monthly_parallel(self, metadata_df: pd.DataFrame,
                                    timestamps: List[datetime],
                                    lat_col: str = 'latitude',
                                    lon_col: str = 'longitude',
                                    station_id_col: str = 'station_id',
                                    force_reprocess: bool = False,
                                    process_months_parallel: bool = True) -> List[str]:
        """
        Extract URMA data with parallel processing at both month and timestamp levels.
        NEW: Uses optimized station index pre-calculation for maximum performance.

        Parameters:
        -----------
        metadata_df : pd.DataFrame
            DataFrame with station metadata
        timestamps : List[datetime]
            List of timestamps to extract
        lat_col : str
            Column name for latitude (default: 'latitude')
        lon_col : str
            Column name for longitude (default: 'longitude')
        station_id_col : str
            Column name for station ID (default: 'station_id')
        force_reprocess : bool
            If True, reprocess even if files exist (default: False)
        process_months_parallel : bool
            If True, process multiple months in parallel (default: True)

        Returns:
        --------
        List[str]
            List of generated filenames
        """

        start_time = time.time()

        # Validate inputs
        required_cols = [lat_col, lon_col, station_id_col]
        missing_cols = [col for col in required_cols if col not in metadata_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Group timestamps by month
        monthly_groups = self._group_timestamps_by_month(timestamps)

        logger.info(f"🚀 Starting optimized parallel processing:")
        logger.info(f"   📊 {len(metadata_df)} stations")
        logger.info(f"   📅 {len(monthly_groups)} months")
        logger.info(f"   ⏰ {len(timestamps)} total timestamps")
        logger.info(f"   📁 Output: {self.output_dir}")
        logger.info(f"   🔄 Timestamp workers: {self.max_workers}")
        logger.info(f"   🔄 Month workers: {self.max_month_workers}")
        logger.info(f"   ⚡ Optimization: Station indices pre-calculated and cached")

        generated_files = []
        skipped_files = []
        failed_files = []

        if process_months_parallel and len(monthly_groups) > 1:
            # Process months in parallel
            logger.info(f"🔄 Processing {len(monthly_groups)} months in parallel...")
            
            # Prepare arguments for parallel month processing
            month_args = [
                (month_key, month_timestamps, metadata_df, lat_col, lon_col, station_id_col, force_reprocess)
                for month_key, month_timestamps in monthly_groups.items()
            ]

            # Create progress tracker for months
            month_progress = ThreadSafeProgressTracker(
                len(monthly_groups), 
                "Month processing"
            )

            try:
                with ThreadPoolExecutor(max_workers=self.max_month_workers) as executor:
                    future_to_month = {
                        executor.submit(self._process_single_month, args): args[0] 
                        for args in month_args
                    }

                    for future in as_completed(future_to_month):
                        month_key = future_to_month[future]
                        year, month = month_key
                        month_name = calendar.month_name[month]
                        
                        try:
                            status, filename = future.result()
                            
                            if status == 'success':
                                generated_files.append(filename)
                                month_progress.update(True, f"✓ {month_name} {year}")
                            elif status == 'skipped':
                                skipped_files.append(filename)
                                month_progress.update(True, f"⏭️ {month_name} {year}")
                            else:
                                failed_files.append(filename)
                                month_progress.update(False, f"✗ {month_name} {year}")

                        except Exception as e:
                            failed_files.append(f"month_{year}_{month}")
                            month_progress.update(False, f"✗ {month_name} {year}")
                            logger.error(f"Error processing {month_name} {year}: {e}")

            finally:
                month_progress.close()

        else:
            # Process months sequentially (but timestamps within each month in parallel)
            for month_key, month_timestamps in monthly_groups.items():
                year, month = month_key
                month_name = calendar.month_name[month]

                logger.info(f"\n📆 Processing {month_name} {year} ({len(month_timestamps)} timestamps)")

                result = self._process_single_month((
                    month_key, month_timestamps, metadata_df, lat_col, lon_col, station_id_col, force_reprocess
                ))
                
                status, filename = result
                
                if status == 'success':
                    generated_files.append(filename)
                elif status == 'skipped':
                    skipped_files.append(filename)
                else:
                    failed_files.append(filename)

        # Final summary
        total_time = time.time() - start_time
        total_processed = len(generated_files) + len(skipped_files) + len(failed_files)
        
        logger.info(f"\n🎉 Optimized parallel processing complete!")
        logger.info(f"⏱️  Total time: {total_time:.1f}s")
        logger.info(f"📄 Generated files: {len(generated_files)}")
        logger.info(f"⏭️  Skipped files: {len(skipped_files)}")
        logger.info(f"❌ Failed files: {len(failed_files)}")
        logger.info(f"⚡ Station indices cached: {len(self._station_indices_cache)} grid combinations")
        
        if total_processed > 0:
            avg_time_per_month = total_time / total_processed
            logger.info(f"📊 Average time per month: {avg_time_per_month:.1f}s")

        if generated_files:
            logger.info(f"\nGenerated files:")
            for filename in generated_files:
                logger.info(f"   ✓ {filename}")

        if skipped_files:
            logger.info(f"\nSkipped files (already exist):")
            for filename in skipped_files:
                logger.info(f"   ⏭️ {filename}")

        if failed_files:
            logger.info(f"\nFailed files:")
            for filename in failed_files:
                logger.info(f"   ❌ {filename}")

        return generated_files

    # Include all the existing methods for backward compatibility
    def extract_data_monthly(self, *args, **kwargs):
        """Backward compatibility wrapper - redirects to parallel version."""
        logger.info("🔄 Using optimized parallel processing (backward compatibility mode)")
        return self.extract_data_monthly_parallel(*args, **kwargs)

    def load_monthly_data(self, year: int = None, month: int = None,
                         load_metadata: bool = True) -> pd.DataFrame:
        """
        Load previously saved monthly data.
        """
        # Find matching files
        pattern = "urma_"
        if year is not None:
            pattern += f"{year}_"
            if month is not None:
                pattern += f"{month:02d}_"

        matching_files = list(self.output_dir.glob(f"{pattern}*.parquet"))

        if not matching_files:
            logger.info(f"No files found matching pattern: {pattern}")
            return pd.DataFrame()

        logger.info(f"📂 Loading {len(matching_files)} files...")

        dataframes = []
        combined_metadata = None

        for filepath in sorted(matching_files):
            try:
                df = pd.read_parquet(filepath)
                dataframes.append(df)
                logger.info(f"   ✓ {filepath.name}: {df.shape[0]} rows")

                # Load metadata from first file
                if load_metadata and combined_metadata is None:
                    metadata_file = filepath.with_suffix('.metadata.json')
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                combined_metadata = json.load(f)
                        except Exception as e:
                            logger.warning(f"Could not load metadata from {metadata_file}: {e}")

            except Exception as e:
                logger.error(f"   ❌ Error loading {filepath.name}: {e}")

        if not dataframes:
            logger.info("No data loaded successfully")
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(dataframes, axis=0).sort_index()

        # Attach metadata if available
        if combined_metadata and load_metadata:
            combined_df.attrs = combined_metadata

        logger.info(f"🎉 Combined shape: {combined_df.shape}")

        return combined_df

    def list_available_files(self):
        """List all available saved files."""
        parquet_files = list(self.output_dir.glob("urma_*.parquet"))

        if not parquet_files:
            logger.info("No saved files found")
            return

        logger.info(f"📁 Available files in {self.output_dir}:")
        logger.info("-" * 80)

        for filepath in sorted(parquet_files):
            try:
                # Get file size
                size_mb = filepath.stat().st_size / (1024 * 1024)

                # Try to get shape info quickly
                df = pd.read_parquet(filepath)
                shape_info = f"{df.shape[0]} rows × {df.shape[1]} cols"

                # Check for metadata file
                metadata_file = filepath.with_suffix('.metadata.json')
                metadata_status = "✓" if metadata_file.exists() else "✗"

                logger.info(f"{filepath.name:<50} | {size_mb:>6.1f} MB | {shape_info} | Meta: {metadata_status}")

            except Exception as e:
                logger.error(f"{filepath.name:<50} | Error: {e}")

def create_timestamp_list(start_time: datetime, end_time: datetime,
                         frequency: str = 'H') -> List[datetime]:
    """Create a list of timestamps between start and end times."""
    time_range = pd.date_range(start=start_time, end=end_time, freq=frequency)
    return time_range.to_pydatetime().tolist()

# Maintain backward compatibility
URMAMonthlyExtractor = URMAMonthlyExtractorParallel

if __name__ == "__main__":
    # Load metadata
    metadata = pd.read_csv('./metadata.csv')

    start_time = datetime(2020, 1, 1, 0)
    end_time = datetime(2025, 7, 15, 23)
    datetime_list = create_timestamp_list(start_time, end_time)

    print(f"Extracting URMA data for {len(datetime_list)} timestamps from {start_time} to {end_time}")

    # Initialize with custom settings for parallel processing
    extractor = URMAMonthlyExtractorParallel(
        output_dir="./",
        max_workers=16,  # Adjust based on your system
        max_month_workers=1  # Process 1 month at a time for optimal memory usage
    )

    # Extract data with optimized parallel processing
    extractor.extract_data_monthly_parallel(
        metadata_df=metadata.reset_index(),
        timestamps=datetime_list,
        process_months_parallel=False  # Can enable for true month-level parallelism
    )