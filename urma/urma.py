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

metadata = pd.read_csv('./metadata.csv')

# Configure logging for notebook use
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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

class URMAMonthlyExtractor:
    """
    Extract point data from NOAA URMA GRIB files for specific station locations.
    Processes and saves data by month with restart capability.
    """

    def __init__(self, output_dir: str = "urma_data"):
        """Initialize the URMA data extractor."""
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.bucket_name = 'noaa-urma-pds'
        self._grid_cache = {}
        
        # NEW: Pre-calculated station indices for each grid configuration
        self._station_indices_cache = {}
        self._grid_stations_initialized = False

        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print(f"✓ URMA Extractor initialized")
        print(f"📁 Output directory: {self.output_dir.absolute()}")

    @contextmanager
    def _fetch_grib_data(self, timestamp: datetime):
        """Fetch and temporarily store GRIB data."""
        temp_file_path = None
        grib_messages = None

        try:
            # Generate S3 key
            date_str = timestamp.strftime('%Y%m%d')
            hour_str = timestamp.strftime('%H')
            s3_key = f"urma2p5.{date_str}/urma2p5.t{hour_str}z.2dvaranl_ndfd.grb2_wexp"

            # Download data
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            grib_data = response['Body'].read()

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.grb2')
            temp_file.write(grib_data)
            temp_file.close()
            temp_file_path = temp_file.name

            # Open with pygrib
            grib_messages = pygrib.open(temp_file_path)
            yield grib_messages

        except Exception as e:
            logger.error(f"Error fetching data for {timestamp}: {e}")
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
        """Get lat/lon grid coordinates with caching."""
        cache_key = f"{grib_message.Ni}_{grib_message.Nj}_{grib_message.gridType}"

        if cache_key not in self._grid_cache:
            print("📍 Generating grid coordinates (one-time setup)...")
            lats, lons = grib_message.latlons()
            self._grid_cache[cache_key] = (lats, lons)
            print(f"✓ Grid cached: {lats.shape}")

        return self._grid_cache[cache_key]

    def _initialize_station_indices(self, station_lats, station_lons, grib_message):
        """
        NEW: Initialize station indices for all grids once at the beginning.
        This replaces the repeated calls to _find_station_indices.
        """
        grid_cache_key = f"{grib_message.Ni}_{grib_message.Nj}_{grib_message.gridType}"
        stations_hash = str(hash(f"{station_lats.tobytes()}{station_lons.tobytes()}"))
        
        # Create combined key for this specific combination of grid and stations
        combined_key = f"{grid_cache_key}_{stations_hash}"
        
        if combined_key in self._station_indices_cache:
            print("✓ Using pre-calculated station indices")
            return self._station_indices_cache[combined_key]

        print(f"🎯 Initializing station indices for {len(station_lats)} stations (one-time setup)...")

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

        # Cache results
        self._station_indices_cache[combined_key] = grid_indices

        print(f"✓ Station indices initialized - Max distance to grid: {distances.max():.4f}°")
        return grid_indices

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
                    'valid_count': int(valid_count),  # Convert to regular int
                    'total_count': int(len(station_values))  # Convert to regular int
                }

            except Exception as e:
                error_msg = f"Message {msg_num} ({getattr(msg, 'name', 'unknown')}): {e}"
                variable_errors.append(error_msg)
                continue

        if variable_errors and len(variable_errors) > 5:
            logger.warning(f"Skipped {len(variable_errors)} variables due to errors")

        return extracted_data

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
            print(f"💾 Saved: {filename} ({df.shape[0]} rows, {df.shape[1]} columns)")

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

    def _extract_month_data(self, metadata_df: pd.DataFrame, timestamps: List[datetime],
                           lat_col: str, lon_col: str, station_id_col: str) -> Optional[pd.DataFrame]:
        """Extract data for a single month's worth of timestamps."""

        # Extract station info
        station_lats = metadata_df[lat_col].values
        station_lons = metadata_df[lon_col].values
        station_ids = metadata_df[station_id_col].tolist()

        all_dataframes = []
        failed_timestamps = []
        variable_metadata = None
        station_indices = None  # NEW: Will be set once and reused

        # Process each timestamp
        for timestamp in tqdm(timestamps, desc="Processing timestamps", leave=False):
            try:
                with self._fetch_grib_data(timestamp) as grib_messages:
                    if grib_messages is None:
                        failed_timestamps.append(timestamp)
                        continue

                    # NEW: Initialize station indices only once for the first successful file
                    if station_indices is None:
                        first_msg = grib_messages.message(1)
                        station_indices = self._initialize_station_indices(
                            station_lats, station_lons, first_msg
                        )
                        print(f"✅ Station indices established - will reuse for all {len(timestamps)} timestamps")

                    # Extract variables using pre-calculated indices
                    grib_messages.rewind()
                    extracted_data = self._extract_variables(grib_messages, station_indices)

                    if not extracted_data:
                        failed_timestamps.append(timestamp)
                        continue

                    # Create DataFrame for this timestamp
                    times = [timestamp] * len(station_ids)
                    index = pd.MultiIndex.from_arrays(
                        [times, station_ids],
                        names=['valid_time', 'station_id']
                    )

                    df_data = {var: info['values'] for var, info in extracted_data.items()}
                    df_single = pd.DataFrame(df_data, index=index)

                    all_dataframes.append(df_single)

                    # Store metadata from first success
                    if variable_metadata is None:
                        variable_metadata = {
                            var: {k: v for k, v in info.items() if k != 'values'}
                            for var, info in extracted_data.items()
                        }

            except Exception as e:
                logger.error(f"Failed processing {timestamp}: {e}")
                failed_timestamps.append(timestamp)

        if not all_dataframes:
            logger.error(f"No data extracted for month")
            return None

        # Combine results for this month
        month_df = pd.concat(all_dataframes, axis=0).sort_index()

        # Add metadata with proper type conversion
        month_df.attrs = {
            'variable_metadata': variable_metadata,
            'source': 'NOAA URMA',
            'extraction_info': {
                'total_timestamps': int(len(timestamps)),
                'successful_timestamps': int(len(all_dataframes)),
                'failed_timestamps': int(len(failed_timestamps)),
                'stations': int(len(station_ids)),
                'month_start': min(timestamps).isoformat(),
                'month_end': max(timestamps).isoformat(),
                'optimization_note': 'Station indices calculated once and reused for all timestamps'
            }
        }

        if failed_timestamps:
            logger.warning(f"Failed {len(failed_timestamps)}/{len(timestamps)} timestamps for this month")

        return month_df

    def extract_data_monthly(self, metadata_df: pd.DataFrame,
                           timestamps: List[datetime],
                           lat_col: str = 'latitude',
                           lon_col: str = 'longitude',
                           station_id_col: str = 'station_id',
                           force_reprocess: bool = False) -> List[str]:
        """
        Extract URMA data for stations and timestamps, processing by month.

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

        # Validate inputs
        required_cols = [lat_col, lon_col, station_id_col]
        missing_cols = [col for col in required_cols if col not in metadata_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Group timestamps by month
        monthly_groups = self._group_timestamps_by_month(timestamps)

        print(f"🚀 Processing {len(metadata_df)} stations across {len(monthly_groups)} months")
        print(f"📅 Date range: {min(timestamps)} to {max(timestamps)}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⚡ Optimization: Station indices will be calculated once per month and reused")

        generated_files = []
        skipped_files = []

        # Process each month
        for month_key, month_timestamps in monthly_groups.items():
            year, month = month_key
            month_name = calendar.month_name[month]

            # Generate filename
            filename = self._generate_filename(year, month, month_timestamps)

            print(f"\n📆 Processing {month_name} {year} ({len(month_timestamps)} timestamps)")

            # Check if file already exists
            if self._file_exists(filename) and not force_reprocess:
                print(f"⏭️  Skipping {filename} (already exists)")
                skipped_files.append(filename)
                continue

            # Extract data for this month
            month_df = self._extract_month_data(
                metadata_df, month_timestamps, lat_col, lon_col, station_id_col
            )

            if month_df is not None:
                # Save the monthly data
                self._save_monthly_data(month_df, filename)
                generated_files.append(filename)

                # Print summary for this month
                success_rate = (month_df.attrs['extraction_info']['successful_timestamps'] /
                              month_df.attrs['extraction_info']['total_timestamps']) * 100
                print(f"✅ {month_name} {year}: {month_df.shape[0]} rows, {success_rate:.1f}% success rate")
            else:
                print(f"❌ Failed to extract data for {month_name} {year}")

        # Final summary
        print(f"\n🎉 Processing complete!")
        print(f"📄 Generated files: {len(generated_files)}")
        print(f"⏭️  Skipped files: {len(skipped_files)}")

        if generated_files:
            print(f"\nGenerated files:")
            for filename in generated_files:
                print(f"   {filename}")

        if skipped_files:
            print(f"\nSkipped files (already exist):")
            for filename in skipped_files:
                print(f"   {filename}")

        return generated_files

    def load_monthly_data(self, year: int = None, month: int = None,
                         load_metadata: bool = True) -> pd.DataFrame:
        """
        Load previously saved monthly data.

        Parameters:
        -----------
        year : int, optional
            Specific year to load (if None, loads all)
        month : int, optional
            Specific month to load (if None, loads all for the year)
        load_metadata : bool
            Whether to load and attach metadata (default: True)

        Returns:
        --------
        pd.DataFrame
            Combined DataFrame from requested files
        """

        # Find matching files
        pattern = "urma_"
        if year is not None:
            pattern += f"{year}_"
            if month is not None:
                pattern += f"{month:02d}_"

        matching_files = list(self.output_dir.glob(f"{pattern}*.parquet"))

        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            return pd.DataFrame()

        print(f"📂 Loading {len(matching_files)} files...")

        dataframes = []
        combined_metadata = None

        for filepath in sorted(matching_files):
            try:
                df = pd.read_parquet(filepath)
                dataframes.append(df)
                print(f"   ✓ {filepath.name}: {df.shape[0]} rows")

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
                print(f"   ❌ Error loading {filepath.name}: {e}")

        if not dataframes:
            print("No data loaded successfully")
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(dataframes, axis=0).sort_index()

        # Attach metadata if available
        if combined_metadata and load_metadata:
            combined_df.attrs = combined_metadata

        print(f"🎉 Combined shape: {combined_df.shape}")

        return combined_df

    def list_available_files(self):
        """List all available saved files."""
        parquet_files = list(self.output_dir.glob("urma_*.parquet"))

        if not parquet_files:
            print("No saved files found")
            return

        print(f"📁 Available files in {self.output_dir}:")
        print("-" * 80)

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

                print(f"{filepath.name:<50} | {size_mb:>6.1f} MB | {shape_info} | Meta: {metadata_status}")

            except Exception as e:
                print(f"{filepath.name:<50} | Error: {e}")

def create_timestamp_list(start_time: datetime, end_time: datetime,
                         frequency: str = 'H') -> List[datetime]:
    """Create a list of timestamps between start and end times."""
    time_range = pd.date_range(start=start_time, end=end_time, freq=frequency)
    return time_range.to_pydatetime().tolist()

if __name__ == "__main__":

    start_time = datetime(2023, 1, 1)
    end_time = datetime(2023, 1, 4, 23)
    datetime_list = create_timestamp_list(start_time, end_time)

    print(f"Extracting URMA data for {len(datetime_list)} timestamps from {start_time} to {end_time}")

    # Initialize with custom output directory
    extractor = URMAMonthlyExtractor(output_dir="./")

    # Extract data by month
    # If script crashes and you restart, it will skip existing files
    # To force reprocessing: force_reprocess=True
    extractor.extract_data_monthly(
        metadata_df=metadata.reset_index(),
        timestamps=datetime_list
    )