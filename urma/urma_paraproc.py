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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import uuid
import pickle

# Configure logging for process-safe operation
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

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

# Global variables for process-shared data
_GLOBAL_STATION_INDICES = None
_GLOBAL_GRID_CACHE = None
_GLOBAL_S3_CLIENT = None

def _get_s3_client():
    """Get process-local S3 client."""
    global _GLOBAL_S3_CLIENT
    if _GLOBAL_S3_CLIENT is None:
        _GLOBAL_S3_CLIENT = boto3.client(
            's3', 
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=50,
                retries={'max_attempts': 3}
            )
        )
    return _GLOBAL_S3_CLIENT

@contextmanager
def _fetch_grib_data(timestamp: datetime, bucket_name: str):
    """Process-safe GRIB data fetching with unique temporary files."""
    temp_file_path = None
    grib_messages = None
    process_id = os.getpid()

    try:
        # Generate S3 key
        date_str = timestamp.strftime('%Y%m%d')
        hour_str = timestamp.strftime('%H')
        s3_key = f"urma2p5.{date_str}/urma2p5.t{hour_str}z.2dvaranl_ndfd.grb2_wexp"

        # Get process-local S3 client
        s3_client = _get_s3_client()

        # Download data
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        grib_data = response['Body'].read()

        # Create unique temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f'_{process_id}_{uuid.uuid4().hex[:8]}.grb2'
        )
        temp_file.write(grib_data)
        temp_file.close()
        temp_file_path = temp_file.name

        # Open with pygrib
        grib_messages = pygrib.open(temp_file_path)
        yield grib_messages

    except Exception as e:
        logger.error(f"Error fetching data for {timestamp} (process {process_id}): {e}")
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

def _get_grid_coordinates(grib_message):
    """Get lat/lon grid coordinates with process-local caching."""
    global _GLOBAL_GRID_CACHE
    
    if _GLOBAL_GRID_CACHE is None:
        _GLOBAL_GRID_CACHE = {}
    
    cache_key = f"{grib_message.Ni}_{grib_message.Nj}_{grib_message.gridType}"
    
    if cache_key not in _GLOBAL_GRID_CACHE:
        logger.debug("📍 Generating grid coordinates (process-local setup)...")
        lats, lons = grib_message.latlons()
        _GLOBAL_GRID_CACHE[cache_key] = (lats, lons)
        logger.info(f"✓ Grid cached in process {os.getpid()}: {lats.shape}")
    
    return _GLOBAL_GRID_CACHE[cache_key]

def _calculate_station_indices(station_lats, station_lons, grib_message):
    """Calculate station indices for the grid."""
    logger.info(f"🎯 Calculating station indices for {len(station_lats)} stations in process {os.getpid()}...")
    
    # Get grid coordinates
    grid_lats, grid_lons = _get_grid_coordinates(grib_message)
    
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
    
    logger.info(f"✓ Station indices calculated - Max distance to grid: {distances.max():.4f}°")
    return grid_indices

def _safe_extract_values(values, station_indices):
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

def _extract_variables(grib_messages, station_indices):
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
                station_values = _safe_extract_values(values, station_indices)

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

def process_single_timestamp(args):
    """
    Process a single timestamp - designed for process-based parallel execution.
    This is a standalone function that can be pickled and sent to worker processes.
    """
    (timestamp, station_lats, station_lons, station_ids, 
     bucket_name, precomputed_indices) = args
    
    try:
        # Set global station indices if provided
        global _GLOBAL_STATION_INDICES
        if precomputed_indices is not None:
            _GLOBAL_STATION_INDICES = precomputed_indices

        with _fetch_grib_data(timestamp, bucket_name) as grib_messages:
            if grib_messages is None:
                return None, timestamp, "Failed to fetch GRIB data"

            # Get or calculate station indices
            if _GLOBAL_STATION_INDICES is not None:
                station_indices = _GLOBAL_STATION_INDICES
            else:
                # Calculate indices if not precomputed
                first_msg = grib_messages.message(1)
                station_indices = _calculate_station_indices(
                    station_lats, station_lons, first_msg
                )
                _GLOBAL_STATION_INDICES = station_indices

            # Extract variables using station indices
            grib_messages.rewind()
            extracted_data = _extract_variables(grib_messages, station_indices)

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

def precompute_station_indices(args):
    """
    Precompute station indices by downloading a sample GRIB file.
    This is a standalone function for process-based execution.
    """
    station_lats, station_lons, sample_timestamps, bucket_name = args
    
    logger.info(f"🔧 Pre-computing station indices in process {os.getpid()}...")
    
    # Try to get a GRIB file to establish the grid
    for timestamp in sample_timestamps[:5]:  # Try first 5 timestamps
        try:
            with _fetch_grib_data(timestamp, bucket_name) as grib_messages:
                if grib_messages is not None:
                    first_msg = grib_messages.message(1)
                    # Calculate and return the station indices
                    station_indices = _calculate_station_indices(
                        station_lats, station_lons, first_msg
                    )
                    logger.info(f"✅ Station indices pre-computed using {timestamp}")
                    return station_indices
        except Exception as e:
            logger.debug(f"Failed to pre-compute with {timestamp}: {e}")
            continue
    
    logger.warning("Could not pre-compute station indices")
    return None

class ProcessSafeProgressTracker:
    """Process-safe progress tracking using shared memory."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.start_time = time.time()
        
        # Use multiprocessing Manager for shared state
        self.manager = mp.Manager()
        self.shared_state = self.manager.dict()
        self.shared_state['completed'] = 0
        self.shared_state['failed'] = 0
        self.lock = self.manager.Lock()
        
        self.pbar = tqdm(total=total_items, desc=description, 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
    def update(self, success: bool = True, message: str = None):
        """Process-safe update of progress."""
        with self.lock:
            if success:
                self.shared_state['completed'] += 1
            else:
                self.shared_state['failed'] += 1
            
            self.pbar.update(1)
            if message:
                self.pbar.set_postfix_str(message)
    
    def close(self):
        """Close progress bar and print summary."""
        self.pbar.close()
        elapsed = time.time() - self.start_time
        completed = self.shared_state['completed']
        success_rate = (completed / self.total_items) * 100
        logger.info(f"{self.description} complete: {completed}/{self.total_items} "
                   f"({success_rate:.1f}%) in {elapsed:.1f}s")

class URMAMonthlyExtractorProcess:
    """
    Process-based parallel URMA data extractor with pre-calculated station indices.
    Uses ProcessPoolExecutor for CPU-intensive operations with true parallelism.
    """

    def __init__(self, output_dir: str = "urma_data", max_workers: int = None, 
                 max_month_workers: int = None):
        """
        Initialize the process-based parallel URMA data extractor.
        
        Parameters:
        -----------
        output_dir : str
            Directory for output files
        max_workers : int
            Maximum workers for timestamp processing (default: cpu_count)
        max_month_workers : int
            Maximum workers for month processing (default: min(4, cpu_count))
        """
        self.bucket_name = 'noaa-urma-pds'

        # Setup parallelism
        cpu_count = mp.cpu_count()
        self.max_workers = max_workers or cpu_count
        self.max_month_workers = max_month_workers or min(4, cpu_count)

        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print(f"✓ Process-based Parallel URMA Extractor initialized")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"🔄 Max workers (timestamps): {self.max_workers}")
        print(f"🔄 Max workers (months): {self.max_month_workers}")
        print(f"💻 Available CPU cores: {cpu_count}")

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

    def _extract_month_data_parallel(self, metadata_df: pd.DataFrame, timestamps: List[datetime],
                                   lat_col: str, lon_col: str, station_id_col: str) -> Optional[pd.DataFrame]:
        """Extract data for a single month using process-based parallel processing."""
        
        start_time = time.time()
        month_name = calendar.month_name[timestamps[0].month]
        year = timestamps[0].year
        
        logger.info(f"🚀 Starting process-based parallel processing for {month_name} {year} "
                   f"({len(timestamps)} timestamps, {self.max_workers} workers)")

        # Extract station info for serialization
        station_lats = metadata_df[lat_col].values
        station_lons = metadata_df[lon_col].values
        station_ids = metadata_df[station_id_col].tolist()

        # Pre-compute station indices in a separate process
        precompute_args = (station_lats, station_lons, timestamps, self.bucket_name)
        
        logger.info("🔧 Pre-computing station indices...")
        with ProcessPoolExecutor(max_workers=1) as precompute_executor:
            future = precompute_executor.submit(precompute_station_indices, precompute_args)
            try:
                precomputed_indices = future.result(timeout=300)  # 5 minute timeout
                if precomputed_indices is not None:
                    logger.info("✅ Station indices pre-computed successfully")
                else:
                    logger.warning("⚠️ Could not pre-compute indices, will calculate per process")
            except Exception as e:
                logger.warning(f"⚠️ Pre-computation failed: {e}, will calculate per process")
                precomputed_indices = None

        # Prepare arguments for parallel processing
        args_list = [
            (ts, station_lats, station_lons, station_ids, self.bucket_name, precomputed_indices)
            for ts in timestamps
        ]

        # Process timestamps in parallel
        all_dataframes = []
        failed_timestamps = []
        variable_metadata = None
        
        # Create progress tracker
        progress = ProcessSafeProgressTracker(
            len(timestamps), 
            f"{month_name} {year} timestamps"
        )

        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_timestamp = {
                    executor.submit(process_single_timestamp, args): args[0] 
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
                'executor_type': 'ProcessPoolExecutor',
                'optimization_note': 'Station indices pre-calculated and distributed to worker processes'
            }
        }

        success_rate = (len(all_dataframes) / len(timestamps)) * 100
        elapsed = time.time() - start_time
        
        logger.info(f"✅ {month_name} {year} complete: {len(all_dataframes)}/{len(timestamps)} "
                   f"({success_rate:.1f}%) in {elapsed:.1f}s")

        return month_df

    def extract_data_monthly_parallel(self, metadata_df: pd.DataFrame,
                                    timestamps: List[datetime],
                                    lat_col: str = 'latitude',
                                    lon_col: str = 'longitude',
                                    station_id_col: str = 'station_id',
                                    force_reprocess: bool = False) -> List[str]:
        """
        Extract URMA data with process-based parallel processing.
        Note: Month-level parallelism is disabled to avoid excessive process creation.

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

        logger.info(f"🚀 Starting process-based parallel processing:")
        logger.info(f"   📊 {len(metadata_df)} stations")
        logger.info(f"   📅 {len(monthly_groups)} months")
        logger.info(f"   ⏰ {len(timestamps)} total timestamps")
        logger.info(f"   📁 Output: {self.output_dir}")
        logger.info(f"   🔄 Process workers: {self.max_workers}")
        logger.info(f"   ⚡ Optimization: Station indices pre-calculated and distributed")

        generated_files = []
        skipped_files = []
        failed_files = []

        # Process months sequentially (timestamps within each month in parallel)
        # Note: Month-level process parallelism would create too many processes
        for month_key, month_timestamps in monthly_groups.items():
            year, month = month_key
            month_name = calendar.month_name[month]

            logger.info(f"\n📆 Processing {month_name} {year} ({len(month_timestamps)} timestamps)")

            # Generate filename
            filename = self._generate_filename(year, month, month_timestamps)

            # Check if file already exists
            if self._file_exists(filename) and not force_reprocess:
                logger.info(f"⏭️  Skipping {filename} (already exists)")
                skipped_files.append(filename)
                continue

            try:
                # Extract data for this month
                month_df = self._extract_month_data_parallel(
                    metadata_df, month_timestamps, lat_col, lon_col, station_id_col
                )

                if month_df is not None:
                    # Save the monthly data
                    self._save_monthly_data(month_df, filename)
                    generated_files.append(filename)
                else:
                    logger.error(f"❌ Failed to extract data for {month_name} {year}")
                    failed_files.append(filename)

            except Exception as e:
                logger.error(f"Error processing {month_name} {year}: {e}")
                failed_files.append(filename)

        # Final summary
        total_time = time.time() - start_time
        total_processed = len(generated_files) + len(skipped_files) + len(failed_files)
        
        logger.info(f"\n🎉 Process-based parallel processing complete!")
        logger.info(f"⏱️  Total time: {total_time:.1f}s")
        logger.info(f"📄 Generated files: {len(generated_files)}")
        logger.info(f"⏭️  Skipped files: {len(skipped_files)}")
        logger.info(f"❌ Failed files: {len(failed_files)}")
        
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

    def extract_data_monthly(self, *args, **kwargs):
        """Backward compatibility wrapper."""
        logger.info("🔄 Using process-based parallel processing")
        return self.extract_data_monthly_parallel(*args, **kwargs)

    def load_monthly_data(self, year: int = None, month: int = None,
                         load_metadata: bool = True) -> pd.DataFrame:
        """Load previously saved monthly data."""
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

# Alias for backward compatibility
URMAMonthlyExtractor = URMAMonthlyExtractorProcess

if __name__ == "__main__":
    # Ensure proper multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Load metadata
    metadata = pd.read_csv('./metadata.csv')

    start_time = datetime(2020, 1, 1, 0)
    end_time = datetime(2025, 7, 15, 23)
    datetime_list = create_timestamp_list(start_time, end_time)

    print(f"Extracting URMA data for {len(datetime_list)} timestamps from {start_time} to {end_time}")

    # Initialize with custom settings for process-based parallel processing
    extractor = URMAMonthlyExtractorProcess(
        output_dir="./",
        max_workers=mp.cpu_count(),  # Use all available CPU cores
        max_month_workers=1  # Process months sequentially to avoid excessive process creation
    )

    # Extract data with process-based parallel processing
    extractor.extract_data_monthly_parallel(
        metadata_df=metadata.reset_index(),
        timestamps=datetime_list
    )