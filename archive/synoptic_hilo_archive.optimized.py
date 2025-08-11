import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import sys
import os
from calendar import monthrange
import glob

class SimpleSynopticDataAPI:
    """
    Simple Synoptic Data API client using the original working approach.
    Includes resume capability to skip already processed months.
    """
    
    def __init__(self, tokens: List[str], max_retries: int = 3, retry_delay: float = 2.0, logger=None):
        """
        Initialize the API client with the original simple approach.
        
        Args:
            tokens (List[str]): List of Synoptic Data API tokens for parallel processing
            max_retries (int): Maximum number of retry attempts for failed chunks
            retry_delay (float): Delay between retry attempts in seconds
            logger: Logger instance for output
        """
        self.tokens = tokens
        self.base_url = "https://api.synopticdata.com/v2/stations/timeseries"
        self.variables = [
            "air_temp",
            "air_temp_2m", 
            "air_temp_high_6_hour",
            "air_temp_low_6_hour",
            "air_temp_high_24_hour",
            "air_temp_low_24_hour"
        ]
        self.lock = threading.Lock()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger or logging.getLogger(__name__)
        
        # Simple stats tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'monthly_files_saved': 0,
            'monthly_files_skipped': 0
        }
        
    def _check_existing_file(self, base_filename: str, month_label: str, min_size_mb: float = 0.1) -> bool:
        """
        Check if a monthly file already exists and has reasonable size.
        
        Args:
            base_filename (str): Base filename pattern
            month_label (str): Month label (YYYY-MM)
            min_size_mb (float): Minimum file size in MB to consider valid
            
        Returns:
            bool: True if file exists and is valid, False otherwise
        """
        filename = f"{base_filename}_{month_label}.csv"
        
        if os.path.exists(filename):
            file_size_mb = os.path.getsize(filename) / (1024**2)
            
            if file_size_mb >= min_size_mb:
                self.logger.info(f"✅ Found existing file: {filename} ({file_size_mb:.1f} MB) - skipping")
                return True
            else:
                self.logger.warning(f"⚠️  Found small file: {filename} ({file_size_mb:.3f} MB) - will reprocess")
                # Optionally backup the small file
                backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    os.rename(filename, backup_name)
                    self.logger.info(f"📦 Backed up small file to: {backup_name}")
                except Exception as e:
                    self.logger.warning(f"⚠️  Could not backup file: {str(e)}")
                return False
        
        return False
    
    def _get_existing_files_info(self, base_filename: str) -> Dict[str, Dict]:
        """
        Get information about all existing files matching the base pattern.
        
        Args:
            base_filename (str): Base filename pattern
            
        Returns:
            Dict: Dictionary with month labels as keys and file info as values
        """
        pattern = f"{base_filename}_*.csv"
        existing_files = glob.glob(pattern)
        
        file_info = {}
        
        for filepath in existing_files:
            filename = os.path.basename(filepath)
            
            # Extract month label from filename
            # Expected format: base_filename_YYYY-MM.csv
            try:
                # Remove base filename and .csv extension
                month_part = filename.replace(f"{os.path.basename(base_filename)}_", "").replace(".csv", "")
                
                # Validate month format (YYYY-MM)
                if len(month_part) == 7 and month_part[4] == '-':
                    year, month = month_part.split('-')
                    if year.isdigit() and month.isdigit() and 1 <= int(month) <= 12:
                        file_size_mb = os.path.getsize(filepath) / (1024**2)
                        file_info[month_part] = {
                            'filepath': filepath,
                            'size_mb': file_size_mb,
                            'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
                        }
            except Exception as e:
                self.logger.warning(f"⚠️  Could not parse filename: {filename} - {str(e)}")
                continue
        
        return file_info
    
    def get_daily_data_chunk(self, cwas: List[str], date: datetime, start_hour: int, 
                           end_hour: int, token: str, chunk_id: str, attempt: int = 1) -> Dict[str, Any]:
        """
        Retrieve data for multiple CWAs and a specific hour range within a day.
        Back to the original working method.
        
        Args:
            cwas (List[str]): List of County Warning Area codes
            date (datetime): Date to retrieve data for
            start_hour (int): Starting hour (0-23)
            end_hour (int): Ending hour (0-23)
            token (str): API token to use for this request
            chunk_id (str): Identifier for this chunk (for logging)
            attempt (int): Current attempt number (for retry logic)
            
        Returns:
            Dict: Combined result with API response data and metadata
        """
        # Format time strings for API
        start_time = date.replace(hour=start_hour, minute=0, second=0).strftime("%Y%m%d%H%M")
        end_time = date.replace(hour=end_hour, minute=59, second=59).strftime("%Y%m%d%H%M")
        
        # Join CWAs with commas (no spaces)
        cwa_string = ",".join(cwas)
        
        params = {
            'token': token,
            'cwa': cwa_string,
            'start': start_time,
            'end': end_time,
            'vars': ','.join(self.variables),
            'network': '1',
            'output': 'json'
        }
        
        try:
            # Add exponential backoff for retries
            if attempt > 1:
                backoff_delay = self.retry_delay * (2 ** (attempt - 2))
                time.sleep(backoff_delay)
            
            # Simple request with reasonable timeout
            response = requests.get(self.base_url, params=params, timeout=90)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if API returned an error
            if 'SUMMARY' in data and data['SUMMARY'].get('RESPONSE_CODE') != 1:
                error_msg = data['SUMMARY'].get('RESPONSE_MESSAGE', 'Unknown error')
                
                if attempt <= self.max_retries:
                    with self.lock:
                        self.logger.warning(f"⚠️  {chunk_id}: API Error (attempt {attempt}/{self.max_retries + 1}) - {error_msg}")
                    return self._retry_chunk(cwas, date, start_hour, end_hour, token, chunk_id, attempt, error_msg)
                else:
                    with self.lock:
                        self.logger.error(f"❌ {chunk_id}: Final failure after {self.max_retries + 1} attempts - {error_msg}")
                    return {
                        'success': False,
                        'data': None,
                        'chunk_id': chunk_id,
                        'error': f"Final failure: {error_msg}",
                        'attempts': attempt
                    }
            
            # Success case
            with self.lock:
                stations_count = len(data.get('STATION', []))
                retry_text = f" (attempt {attempt})" if attempt > 1 else ""
                self.logger.info(f"✅ {chunk_id}: {stations_count} stations retrieved{retry_text}")
                self.stats['successful_requests'] += 1
            
            return {
                'success': True,
                'data': data,
                'chunk_id': chunk_id,
                'cwas': cwas,
                'date': date,
                'attempts': attempt
            }
            
        except requests.exceptions.RequestException as e:
            if attempt <= self.max_retries:
                with self.lock:
                    self.logger.warning(f"⚠️  {chunk_id}: Request error (attempt {attempt}/{self.max_retries + 1}) - {str(e)}")
                return self._retry_chunk(cwas, date, start_hour, end_hour, token, chunk_id, attempt, str(e))
            else:
                with self.lock:
                    self.logger.error(f"❌ {chunk_id}: Final failure after {self.max_retries + 1} attempts - {str(e)}")
                    self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'data': None,
                    'chunk_id': chunk_id,
                    'error': f"Final failure: {str(e)}",
                    'attempts': attempt
                }
        except json.JSONDecodeError as e:
            if attempt <= self.max_retries:
                with self.lock:
                    self.logger.warning(f"⚠️  {chunk_id}: JSON decode error (attempt {attempt}/{self.max_retries + 1}) - {str(e)}")
                return self._retry_chunk(cwas, date, start_hour, end_hour, token, chunk_id, attempt, str(e))
            else:
                with self.lock:
                    self.logger.error(f"❌ {chunk_id}: Final failure after {self.max_retries + 1} attempts - {str(e)}")
                    self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'data': None,
                    'chunk_id': chunk_id,
                    'error': f"Final failure: {str(e)}",
                    'attempts': attempt
                }
    
    def _retry_chunk(self, cwas: List[str], date: datetime, start_hour: int, 
                    end_hour: int, token: str, chunk_id: str, attempt: int, error: str) -> Dict[str, Any]:
        """
        Helper method to retry a failed chunk with a different token if available.
        
        Returns:
            Dict: Result of retry attempt
        """
        # Try a different token for retry if available
        if len(self.tokens) > 1:
            # Use next token in rotation
            token_index = (self.tokens.index(token) + 1) % len(self.tokens)
            retry_token = self.tokens[token_index]
        else:
            retry_token = token
        
        return self.get_daily_data_chunk(cwas, date, start_hour, end_hour, retry_token, chunk_id, attempt + 1)
    
    def parse_station_data(self, api_response: Dict[str, Any], cwas: List[str], date: datetime) -> pd.DataFrame:
        """
        Parse API response into a pandas DataFrame with separate columns for each variable.
        Using the original parsing method.
        
        Args:
            api_response (Dict): Raw API response
            cwas (List[str]): List of County Warning Area codes queried
            date (datetime): Date of the data
            
        Returns:
            pd.DataFrame: Parsed weather data with separate columns for each variable
        """
        if not api_response or 'STATION' not in api_response:
            return pd.DataFrame()
        
        all_records = []
        
        for station in api_response['STATION']:
            station_id = station.get('STID', 'Unknown')
            station_name = station.get('NAME', 'Unknown')
            latitude = station.get('LATITUDE', None)
            longitude = station.get('LONGITUDE', None)
            elevation = station.get('ELEVATION', None)
            
            if 'OBSERVATIONS' not in station:
                continue
                
            observations = station['OBSERVATIONS']
            
            # Get timestamps
            if 'date_time' not in observations:
                continue
                
            timestamps = observations['date_time']
            
            # Handle both single values and arrays for timestamps
            if not isinstance(timestamps, list):
                timestamps = [timestamps]
            
            # Create a record for each timestamp
            for i, timestamp in enumerate(timestamps):
                record = {
                    'station_id': station_id,
                    'station_name': station_name,
                    'latitude': latitude,
                    'longitude': longitude,
                    'elevation': elevation,
                    'datetime': pd.to_datetime(timestamp)
                }
                
                # Add each variable as a separate column
                for base_var in self.variables:
                    var_key = f"{base_var}_set_1"
                    
                    if var_key in observations:
                        values = observations[var_key]
                        
                        # Handle both single values and arrays
                        if not isinstance(values, list):
                            values = [values]
                        
                        # Get the value for this timestamp index, or None if not available
                        if i < len(values):
                            record[base_var] = values[i]
                        else:
                            record[base_var] = None
                    else:
                        # Variable not present for this station
                        record[base_var] = None
                
                all_records.append(record)
        
        return pd.DataFrame(all_records)
    
    def _get_month_date_ranges(self, start_date: str, end_date: str) -> List[Dict[str, str]]:
        """
        Generate monthly date ranges from start to end date.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            List[Dict]: List of monthly date ranges with 'start', 'end', and 'month_label'
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        monthly_ranges = []
        current_date = start_dt
        
        while current_date <= end_dt:
            # Calculate month boundaries
            year = current_date.year
            month = current_date.month
            
            # First day of current month
            month_start = datetime(year, month, 1)
            
            # Last day of current month
            last_day = monthrange(year, month)[1]
            month_end = datetime(year, month, last_day)
            
            # Use actual start/end dates if they fall within this month
            range_start = max(start_dt, month_start)
            range_end = min(end_dt, month_end)
            
            month_label = f"{year:04d}-{month:02d}"
            
            monthly_ranges.append({
                'start': range_start.strftime('%Y-%m-%d'),
                'end': range_end.strftime('%Y-%m-%d'),
                'month_label': month_label,
                'year': year,
                'month': month
            })
            
            # Move to next month
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)
        
        return monthly_ranges
    
    def _save_monthly_data(self, monthly_data: pd.DataFrame, month_label: str, base_filename: str) -> str:
        """
        Save monthly data to a CSV file.
        
        Args:
            monthly_data (pd.DataFrame): Data for the month
            month_label (str): Month label (YYYY-MM)
            base_filename (str): Base filename pattern
            
        Returns:
            str: Full filename of saved file
        """
        if monthly_data.empty:
            self.logger.warning(f"⚠️  No data to save for month {month_label}")
            return ""
        
        # Create filename with month
        filename = f"{base_filename}_{month_label}.csv"
        
        # Sort data before saving
        if 'datetime' in monthly_data.columns and 'station_id' in monthly_data.columns:
            monthly_data_sorted = monthly_data.set_index(['datetime', 'station_id']).sort_index()
        else:
            monthly_data_sorted = monthly_data.sort_values(['datetime', 'station_id']) if 'datetime' in monthly_data.columns else monthly_data
        
        # Save to CSV
        monthly_data_sorted.to_csv(filename, index=True)
        
        # Update stats
        with self.lock:
            self.stats['monthly_files_saved'] += 1
        
        file_size = os.path.getsize(filename) / (1024**2)
        self.logger.info(f"💾 Monthly data saved: {filename} ({len(monthly_data):,} records, {file_size:.1f} MB)")
        return filename
    
    def get_data_range_monthly(self, cwas: List[str], start_date: str, end_date: str, 
                              hour_chunk: int = 24, max_workers: int = None, 
                              force_reprocess: bool = False) -> List[str]:
        """
        Retrieve data for date range in time chunks using parallel processing.
        Process and save each month separately, with resume capability.
        
        Args:
            cwas (List[str]): List of County Warning Area codes
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            hour_chunk (int): Hours per chunk (24 for daily, 12 for 12-hour, 6 for 6-hour chunks)
            max_workers (int): Maximum number of parallel workers (defaults to number of tokens)
            force_reprocess (bool): If True, reprocess all months even if files exist
            
        Returns:
            List[str]: List of saved filenames
        """
        # Get monthly date ranges
        monthly_ranges = self._get_month_date_ranges(start_date, end_date)
        
        # Create base filename
        base_filename = f"simple_synoptic_data_{start_date}_to_{end_date}"
        
        # Check for existing files if not forcing reprocess
        existing_files_info = {}
        months_to_process = []
        months_to_skip = []
        
        if not force_reprocess:
            existing_files_info = self._get_existing_files_info(base_filename)
            
            for month_range in monthly_ranges:
                month_label = month_range['month_label']
                if self._check_existing_file(base_filename, month_label):
                    months_to_skip.append(month_range)
                    with self.lock:
                        self.stats['monthly_files_skipped'] += 1
                else:
                    months_to_process.append(month_range)
        else:
            months_to_process = monthly_ranges
            self.logger.info("🔄 Force reprocess enabled - will reprocess all months")
        
        self.logger.info(f"🚀 SIMPLE MONTHLY DATA RETRIEVAL STARTED (WITH RESUME)")
        self.logger.info(f"📊 CWAs: {', '.join(cwas)}")
        self.logger.info(f"📅 Date range: {start_date} to {end_date}")
        self.logger.info(f"📆 Total months: {len(monthly_ranges)}")
        self.logger.info(f"✅ Months to skip (already done): {len(months_to_skip)}")
        self.logger.info(f"🔄 Months to process: {len(months_to_process)}")
        self.logger.info(f"🔑 Available tokens: {len(self.tokens)}")
        
        # List existing files
        if existing_files_info:
            self.logger.info(f"\n📂 Found existing files:")
            for month_label, info in sorted(existing_files_info.items()):
                self.logger.info(f"  • {month_label}: {info['size_mb']:.1f} MB (modified: {info['modified'].strftime('%Y-%m-%d %H:%M')})")
        
        # List months being skipped
        if months_to_skip:
            self.logger.info(f"\n⏭️  Skipping months:")
            for month_range in months_to_skip:
                self.logger.info(f"  • {month_range['month_label']} ({month_range['start']} to {month_range['end']})")
        
        # List months to process
        if months_to_process:
            self.logger.info(f"\n🔄 Processing months:")
            for month_range in months_to_process:
                self.logger.info(f"  • {month_range['month_label']} ({month_range['start']} to {month_range['end']})")
        else:
            self.logger.info(f"\n🎉 All months already completed! Nothing to process.")
            # Return list of existing files
            existing_files = [info['filepath'] for info in existing_files_info.values()]
            return existing_files
        
        saved_files = []
        total_records = 0
        overall_start_time = time.time()
        
        # Process each month that needs processing
        for i, month_range in enumerate(months_to_process, 1):
            month_label = month_range['month_label']
            month_start = month_range['start']
            month_end = month_range['end']
            
            self.logger.info(f"\n🗓️  Processing month {i}/{len(months_to_process)}: {month_label}")
            self.logger.info(f"📅 Month range: {month_start} to {month_end}")
            self.logger.info("-" * 60)
            
            try:
                # Get data for this month using original method
                monthly_data = self._get_month_data_original(
                    cwas=cwas,
                    start_date=month_start,
                    end_date=month_end,
                    hour_chunk=hour_chunk,
                    max_workers=max_workers
                )
                
                # Save monthly data
                if not monthly_data.empty:
                    filename = self._save_monthly_data(monthly_data, month_label, base_filename)
                    if filename:
                        saved_files.append(filename)
                        total_records += len(monthly_data)
                else:
                    self.logger.warning(f"⚠️  No data retrieved for month {month_label}")
                
                # Clear monthly data from memory
                del monthly_data
                
            except Exception as e:
                self.logger.error(f"❌ Error processing month {month_label}: {str(e)}")
                continue
        
        # Add existing files to the list
        for info in existing_files_info.values():
            if os.path.exists(info['filepath']):
                saved_files.append(info['filepath'])
        
        # Final summary
        total_time = time.time() - overall_start_time
        
        self.logger.info(f"\n" + "=" * 80)
        self.logger.info(f"🎉 SIMPLE MONTHLY DATA RETRIEVAL COMPLETE!")
        self.logger.info(f"=" * 80)
        self.logger.info(f"📁 Total files available: {len(saved_files)}")
        self.logger.info(f"💾 Files processed this run: {self.stats['monthly_files_saved']}")
        self.logger.info(f"⏭️  Files skipped (already existed): {self.stats['monthly_files_skipped']}")
        self.logger.info(f"📊 New records processed: {total_records:,}")
        self.logger.info(f"⏱️  Processing time: {total_time:.2f} seconds")
        self.logger.info(f"📡 Total requests: {self.stats['total_requests']}")
        self.logger.info(f"✅ Successful requests: {self.stats['successful_requests']}")
        self.logger.info(f"❌ Failed requests: {self.stats['failed_requests']}")
        
        # List all available files
        if saved_files:
            self.logger.info(f"\n📂 All available files:")
            for filename in sorted(saved_files):
                if os.path.exists(filename):
                    file_size = os.path.getsize(filename) / (1024**2)  # MB
                    status = "NEW" if filename in [f for f in saved_files if f not in [info['filepath'] for info in existing_files_info.values()]] else "EXISTING"
                    self.logger.info(f"  • {os.path.basename(filename)} ({file_size:.1f} MB) [{status}]")
        
        return saved_files
    
    def _get_month_data_original(self, cwas: List[str], start_date: str, end_date: str, 
                                hour_chunk: int = 24, max_workers: int = None) -> pd.DataFrame:
        """
        Get data for a single month using the original working method.
        
        Args:
            cwas (List[str]): List of County Warning Area codes
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            hour_chunk (int): Hours per chunk
            max_workers (int): Maximum number of parallel workers
            
        Returns:
            pd.DataFrame: Combined weather data for the month
        """
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Calculate total chunks
        chunks_per_day = 24 // hour_chunk
        total_days = (end_dt - start_dt).days + 1
        total_chunks = total_days * chunks_per_day
        
        # Set max workers
        if max_workers is None:
            max_workers = min(len(self.tokens), total_chunks)
        
        self.logger.info(f"🔢 Month chunks: {total_chunks}")
        self.logger.info(f"🧵 Parallel workers: {max_workers}")
        
        # Create list of all chunks to process
        chunk_tasks = []
        current_date = start_dt
        chunk_counter = 1
        
        while current_date <= end_dt:
            for chunk in range(chunks_per_day):
                start_hour = chunk * hour_chunk
                end_hour = start_hour + hour_chunk - 1
                
                chunk_id = f"Chunk-{chunk_counter:03d} ({current_date.strftime('%Y-%m-%d')} {start_hour:02d}:00-{end_hour:02d}:59)"
                
                chunk_tasks.append({
                    'cwas': cwas,
                    'date': current_date,
                    'start_hour': start_hour,
                    'end_hour': end_hour,
                    'chunk_id': chunk_id
                })
                
                chunk_counter += 1
            
            current_date += timedelta(days=1)
        
        # Process chunks in parallel using original method
        all_data = []
        successful_chunks = 0
        failed_chunks = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {}
            
            for i, chunk_task in enumerate(chunk_tasks):
                # Cycle through available tokens
                token = self.tokens[i % len(self.tokens)]
                
                future = executor.submit(
                    self.get_daily_data_chunk,
                    chunk_task['cwas'],
                    chunk_task['date'],
                    chunk_task['start_hour'],
                    chunk_task['end_hour'],
                    token,
                    chunk_task['chunk_id']
                )
                future_to_chunk[future] = chunk_task
            
            # Process completed futures
            for future in as_completed(future_to_chunk):
                chunk_task = future_to_chunk[future]
                
                try:
                    result = future.result()
                    
                    with self.lock:
                        self.stats['total_requests'] += 1
                    
                    if result['success']:
                        # Parse the data
                        df_chunk = self.parse_station_data(
                            result['data'], 
                            result['cwas'], 
                            result['date']
                        )
                        
                        if not df_chunk.empty:
                            all_data.append(df_chunk)
                            successful_chunks += 1
                        else:
                            with self.lock:
                                self.logger.warning(f"⚠️  {result['chunk_id']}: No data in response")
                    else:
                        failed_chunks += 1
                        with self.lock:
                            self.logger.error(f"💀 {result['chunk_id']}: PERMANENTLY FAILED - {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    with self.lock:
                        self.logger.error(f"❌ {chunk_task['chunk_id']}: Exception - {str(e)}")
                    failed_chunks += 1
                
                # Progress update
                completed = successful_chunks + failed_chunks
                if completed % 10 == 0:  # Update every 10 chunks
                    progress = (completed / total_chunks) * 100
                    
                    with self.lock:
                        self.logger.info(f"📈 Progress: {completed}/{total_chunks} ({progress:.1f}%) | ✅ {successful_chunks} | ❌ {failed_chunks}")
        
        # Combine all data
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            self.logger.info(f"✅ Month complete: {len(final_df):,} records")
            self.logger.info(f"📊 Success rate: {(successful_chunks/total_chunks)*100:.1f}%")
            
            return final_df
        else:
            self.logger.warning(f"⚠️  No data retrieved for this month")
            return pd.DataFrame()


def setup_logging(log_filename: str = None) -> logging.Logger:
    """Set up logging configuration."""
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"simple_synoptic_resume_{timestamp}.log"
    
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Simple processing log (with resume): {log_path}")
    
    return logger


def main():
    """Main function using the original simple approach with resume capability."""
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python simple_synoptic_monthly_logger_resume.py <start_date> <end_date> [--force]")
        print("Example: python simple_synoptic_monthly_logger_resume.py 2020-01-01 2021-01-01")
        print("         python simple_synoptic_monthly_logger_resume.py 2020-01-01 2021-01-01 --force")
        print("")
        print("Options:")
        print("  --force    Reprocess all months even if CSV files already exist")
        sys.exit(1)
    
    START_DATE = sys.argv[1]
    END_DATE = sys.argv[2]
    FORCE_REPROCESS = len(sys.argv) == 4 and sys.argv[3] == '--force'
    
    # Validate dates
    try:
        start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
        
        if start_dt > end_dt:
            print("Error: Start date must be before end date")
            sys.exit(1)
            
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging()
    
    # Configuration - same as original
    API_TOKENS = [
        'a2386b75ecbc4c2784db1270695dde73',			
        'ecd8cc8856884bcc8f02f374f8eb87fc',			
        '97985ec5837949cf807aa36544a7ca57',			
        '1445303bab134661bdae8e1155482ff0',			
        '265e9a7f586d45219a63a7afbe256b33',			
        '2eb808b92c9841adb31e2b2608e9afc3',			
        'ee3af9f0107142f98764d24fbfff3348',			
        '31eaaf2ef5cc40519af576ba2bd0db65',			
        'b5ee3feab5fc44f3a0a7ec254faf3ebe',			
        'fbebef61121547fa9b1f1a066073b693',			
        '362a6b17afd246848d5fa7395fbec5ad',			
        'db846bc581fc4ee1a92b10dc78eb73cb',			
        '52ac84fcd70f4610a535f9fedef85d67',			
        '10420587e65c4ebd94ea6cd1e7f8d48a',			
        '42bd9506637645e5990b15710913f36d',			
        '96b09438b9564b1b9f7ba12c27724bd4',			
        'f29e8128459b4e3bb8017a4f44e411ca',			
        '11a0601a70154e21b48aa8a5b3d1f46d',			
        '6e250065549b4364b1649a17011da08a',			
        '0b4c54bbebb1462182bf8f0aee8635cd',
    ]
    
    CWA_LIST = ["BYZ", "BOI", "LKN", "EKA", "FGZ", "GGW", "TFX", "VEF", "LOX", "MFR",
                "MSO", "PDT", "PSR", "PIH", "PQR", "REV", "STO", "SLC", "SGX", "MTR",
                "HNX", "SEW", "OTX", "TWC"]
    
    # Time chunk size and retry configuration - same as original
    HOUR_CHUNK = 24
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    MAX_WORKERS = None
    
    # Log script start
    logger.info(f"🚀 Simple Synoptic Data Retrieval Script Started (WITH RESUME)")
    logger.info(f"📅 Date range: {START_DATE} to {END_DATE}")
    logger.info(f"🔄 Force reprocess: {FORCE_REPROCESS}")
    logger.info(f"🔐 Process ID: {os.getpid()}")
    logger.info(f"📂 Working directory: {os.getcwd()}")
    logger.info(f"👤 User: {os.getenv('USER', 'unknown')}")
    
    try:
        # Initialize API client with original simple approach
        api_client = SimpleSynopticDataAPI(
            API_TOKENS, 
            max_retries=MAX_RETRIES, 
            retry_delay=RETRY_DELAY, 
            logger=logger
        )
        
        # Retrieve data with monthly saving and resume capability
        start_time = time.time()
        saved_files = api_client.get_data_range_monthly(
            cwas=CWA_LIST,
            start_date=START_DATE,
            end_date=END_DATE,
            hour_chunk=HOUR_CHUNK,
            max_workers=MAX_WORKERS,
            force_reprocess=FORCE_REPROCESS
        )
        end_time = time.time()
        
        if saved_files:
            # Performance metrics
            total_time = end_time - start_time
            logger.info(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
            logger.info(f"📁 Total files available: {len(saved_files)}")
            
            logger.info(f"\n🎉 SIMPLE PROCESSING WITH RESUME COMPLETE!")
            
        else:
            logger.error("❌ No files were available.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 SCRIPT FAILED: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)
    
    finally:
        logger.info(f"🔚 Script execution ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()