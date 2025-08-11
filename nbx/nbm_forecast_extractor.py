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

class NBMForecastExtractor:
    """
    Extract point data from NOAA NBM GRIB files for specific station locations.
    Processes and saves forecast data by month with restart capability.
    """

    def __init__(self, output_dir: str = "nbm_data"):
        """Initialize the NBM forecast data extractor."""
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.bucket_name = 'noaa-nbm-grib2-pds'
        self._grid_cache = {}
        
        # Pre-calculated station indices for each grid configuration
        self._station_indices_cache = {}
        self._grid_stations_initialized = False

        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print(f"✓ NBM Forecast Extractor initialized")
        print(f"📁 Output directory: {self.output_dir.absolute()}")

    @contextmanager
    def _fetch_grib_data(self, forecast_time: datetime, init_time: str, forecast_hour: str):
        """Fetch and temporarily store GRIB data."""
        temp_file_path = None
        grib_messages = None

        try:
            # Generate S3 key
            date_str = forecast_time.strftime('%Y%m%d')
            s3_key = f"blend.{date_str}/{init_time}/core/blend.t{init_time}z.core.f{forecast_hour}.co.grib2"

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
            logger.error(f"Error fetching data for {forecast_time} t{init_time}z f{forecast_hour}: {e}")
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
        Initialize station indices for all grids once at the beginning.
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

    def _generate_forecast_combinations(self, start_date: datetime, end_date: datetime, 
                                      init_times: List[str], forecast_hours: List[str]) -> List[Tuple[datetime, str, str, datetime]]:
        """Generate all combinations of forecast dates, init times, and forecast hours."""
        combinations = []
        
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            for init_time in init_times:
                for forecast_hour in forecast_hours:
                    # Calculate valid time (forecast time + forecast hour)
                    forecast_time = datetime.combine(current_date, datetime.min.time())
                    forecast_time = forecast_time.replace(hour=int(init_time))
                    
                    valid_time = forecast_time + timedelta(hours=int(forecast_hour))
                    
                    combinations.append((forecast_time, init_time, forecast_hour, valid_time))
            
            current_date += timedelta(days=1)
        
        return combinations

    def _group_forecasts_by_month(self, forecast_combinations: List[Tuple[datetime, str, str, datetime]]) -> Dict[Tuple[int, int], List[Tuple[datetime, str, str, datetime]]]:
        """Group forecast combinations by (year, month) of the forecast time."""
        monthly_groups = {}

        for combo in forecast_combinations:
            forecast_time = combo[0]
            month_key = (forecast_time.year, forecast_time.month)
            if month_key not in monthly_groups:
                monthly_groups[month_key] = []
            monthly_groups[month_key].append(combo)

        # Sort combinations within each month
        for month_key in monthly_groups:
            monthly_groups[month_key].sort(key=lambda x: (x[0], x[1], x[2]))  # Sort by forecast_time, init_time, forecast_hour

        return monthly_groups

    def _generate_filename(self, year: int, month: int, forecast_combinations: List[Tuple[datetime, str, str, datetime]]) -> str:
        """Generate filename based on year, month, and forecast range."""
        first_forecast = min(combo[0] for combo in forecast_combinations).strftime('%Y%m%d')
        last_forecast = max(combo[0] for combo in forecast_combinations).strftime('%Y%m%d')
        month_name = calendar.month_abbr[month].lower()

        return f"nbm_{year}_{month:02d}_{month_name}_{first_forecast}_to_{last_forecast}.parquet"

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

    def _extract_month_data(self, metadata_df: pd.DataFrame, forecast_combinations: List[Tuple[datetime, str, str, datetime]],
                           lat_col: str, lon_col: str, station_id_col: str) -> Optional[pd.DataFrame]:
        """Extract data for a single month's worth of forecast combinations."""

        # Extract station info
        station_lats = metadata_df[lat_col].values
        station_lons = metadata_df[lon_col].values
        station_ids = metadata_df[station_id_col].tolist()

        all_dataframes = []
        failed_forecasts = []
        variable_metadata = None
        station_indices = None

        # Process each forecast combination
        for forecast_time, init_time, forecast_hour, valid_time in tqdm(forecast_combinations, desc="Processing forecasts", leave=False):
            try:
                with self._fetch_grib_data(forecast_time, init_time, forecast_hour) as grib_messages:
                    if grib_messages is None:
                        failed_forecasts.append((forecast_time, init_time, forecast_hour, valid_time))
                        continue

                    # Initialize station indices only once for the first successful file
                    if station_indices is None:
                        first_msg = grib_messages.message(1)
                        station_indices = self._initialize_station_indices(
                            station_lats, station_lons, first_msg
                        )
                        print(f"✅ Station indices established - will reuse for all {len(forecast_combinations)} forecasts")

                    # Extract variables using pre-calculated indices
                    grib_messages.rewind()
                    extracted_data = self._extract_variables(grib_messages, station_indices)

                    if not extracted_data:
                        failed_forecasts.append((forecast_time, init_time, forecast_hour, valid_time))
                        continue

                    # Create DataFrame for this forecast
                    forecast_times = [forecast_time] * len(station_ids)
                    valid_times = [valid_time] * len(station_ids)
                    index = pd.MultiIndex.from_arrays(
                        [forecast_times, valid_times, station_ids],
                        names=['forecast_time', 'valid_time', 'station_id']
                    )

                    df_data = {var: info['values'] for var, info in extracted_data.items()}
                    df_single = pd.DataFrame(df_data, index=index)

                    # Add forecast metadata as columns
                    df_single['init_time'] = init_time
                    df_single['forecast_hour'] = int(forecast_hour)

                    all_dataframes.append(df_single)

                    # Store metadata from first success
                    if variable_metadata is None:
                        variable_metadata = {
                            var: {k: v for k, v in info.items() if k != 'values'}
                            for var, info in extracted_data.items()
                        }

            except Exception as e:
                logger.error(f"Failed processing {forecast_time} t{init_time}z f{forecast_hour}: {e}")
                failed_forecasts.append((forecast_time, init_time, forecast_hour, valid_time))

        if not all_dataframes:
            logger.error(f"No data extracted for month")
            return None

        # Combine results for this month
        month_df = pd.concat(all_dataframes, axis=0).sort_index()

        # Add metadata with proper type conversion
        month_df.attrs = {
            'variable_metadata': variable_metadata,
            'source': 'NOAA NBM',
            'extraction_info': {
                'total_forecasts': int(len(forecast_combinations)),
                'successful_forecasts': int(len(all_dataframes)),
                'failed_forecasts': int(len(failed_forecasts)),
                'stations': int(len(station_ids)),
                'month_start': min(combo[0] for combo in forecast_combinations).isoformat(),
                'month_end': max(combo[0] for combo in forecast_combinations).isoformat(),
                'optimization_note': 'Station indices calculated once and reused for all forecasts'
            }
        }

        if failed_forecasts:
            logger.warning(f"Failed {len(failed_forecasts)}/{len(forecast_combinations)} forecasts for this month")

        return month_df

    def extract_forecast_data_monthly(self, metadata_df: pd.DataFrame,
                                    start_date: datetime,
                                    end_date: datetime,
                                    init_times: List[str],
                                    forecast_hours: List[str],
                                    lat_col: str = 'latitude',
                                    lon_col: str = 'longitude',
                                    station_id_col: str = 'station_id',
                                    force_reprocess: bool = False) -> List[str]:
        """
        Extract NBM forecast data for stations and date range, processing by month.

        Parameters:
        -----------
        metadata_df : pd.DataFrame
            DataFrame with station metadata
        start_date : datetime
            Start date for forecast extraction
        end_date : datetime
            End date for forecast extraction
        init_times : List[str]
            List of initialization times (e.g., ['00', '12'] for t00z and t12z)
        forecast_hours : List[str]
            List of forecast hours (e.g., ['024', '048', '072'])
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

        # Generate all forecast combinations
        forecast_combinations = self._generate_forecast_combinations(start_date, end_date, init_times, forecast_hours)
        
        # Group forecast combinations by month
        monthly_groups = self._group_forecasts_by_month(forecast_combinations)

        print(f"🚀 Processing {len(metadata_df)} stations across {len(monthly_groups)} months")
        print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        print(f"🕐 Init times: {init_times}")
        print(f"⏰ Forecast hours: {forecast_hours}")
        print(f"📊 Total forecast combinations: {len(forecast_combinations)}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⚡ Optimization: Station indices will be calculated once per month and reused")

        generated_files = []
        skipped_files = []

        # Process each month
        for month_key, month_forecasts in monthly_groups.items():
            year, month = month_key
            month_name = calendar.month_name[month]

            # Generate filename
            filename = self._generate_filename(year, month, month_forecasts)

            print(f"\n📆 Processing {month_name} {year} ({len(month_forecasts)} forecast combinations)")

            # Check if file already exists
            if self._file_exists(filename) and not force_reprocess:
                print(f"⏭️  Skipping {filename} (already exists)")
                skipped_files.append(filename)
                continue

            # Extract data for this month
            month_df = self._extract_month_data(
                metadata_df, month_forecasts, lat_col, lon_col, station_id_col
            )

            if month_df is not None:
                # Save the monthly data
                self._save_monthly_data(month_df, filename)
                generated_files.append(filename)

                # Print summary for this month
                success_rate = (month_df.attrs['extraction_info']['successful_forecasts'] /
                              month_df.attrs['extraction_info']['total_forecasts']) * 100
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
        Load previously saved monthly forecast data.

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
        pattern = "nbm_"
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
        parquet_files = list(self.output_dir.glob("nbm_*.parquet"))

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

if __name__ == "__main__":
    # Example usage
    
    # Load station metadata (you'll need to provide your own metadata.csv)
    # metadata = pd.read_csv('./metadata.csv')
    
    # Example configuration
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 3)
    init_times = ['00', '12']  # t00z and t12z
    forecast_hours = ['024', '048', '072']  # 24, 48, and 72 hour forecasts

    print(f"Extracting NBM forecast data from {start_date.date()} to {end_date.date()}")
    print(f"Init times: {init_times}")
    print(f"Forecast hours: {forecast_hours}")

    # Initialize extractor with custom output directory
    extractor = NBMForecastExtractor(output_dir="./nbm_data")

    # Extract forecast data by month
    # Uncomment and modify the following lines once you have your metadata file:
    
    # extractor.extract_forecast_data_monthly(
    #     metadata_df=metadata.reset_index(),
    #     start_date=start_date,
    #     end_date=end_date,
    #     init_times=init_times,
    #     forecast_hours=forecast_hours
    # )
    
    print("NBM Forecast Extractor is ready to use!")
    print("Please ensure you have a metadata.csv file with station information before running extraction.")