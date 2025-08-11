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

class SynopticDataAPIParallel:
    """
    A parallelized class to interact with the Synoptic Data API for weather data retrieval.
    Uses multiple API tokens to parallelize requests across time chunks with retry logic.
    """
    
    def __init__(self, tokens: List[str], max_retries: int = 3, retry_delay: float = 2.0, logger=None):
        """
        Initialize the API client with multiple authentication tokens.
        
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
        
    def get_daily_data_chunk(self, cwas: List[str], date: datetime, start_hour: int, 
                           end_hour: int, token: str, chunk_id: str, attempt: int = 1) -> Dict[str, Any]:
        """
        Retrieve data for multiple CWAs and a specific hour range within a day.
        
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
            
            response = requests.get(self.base_url, params=params, timeout=60)
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
    
    def get_data_range(self, cwas: List[str], start_date: str, end_date: str, 
                       hour_chunk: int = 24, max_workers: int = None) -> pd.DataFrame:
        """
        Retrieve data for multiple CWAs over a date range in time chunks using parallel processing with retry logic.
        
        Args:
            cwas (List[str]): List of County Warning Area codes
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            hour_chunk (int): Hours per chunk (24 for daily, 12 for 12-hour, 6 for 6-hour chunks)
            max_workers (int): Maximum number of parallel workers (defaults to number of tokens)
            
        Returns:
            pd.DataFrame: Combined weather data for all CWAs and dates
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
        
        self.logger.info(f"🚀 PARALLEL DATA RETRIEVAL STARTED (with retry logic)")
        self.logger.info(f"📊 CWAs: {', '.join(cwas)}")
        self.logger.info(f"📅 Date range: {start_date} to {end_date} ({total_days} days)")
        self.logger.info(f"⏰ Time chunks: {hour_chunk}-hour periods ({chunks_per_day} per day)")
        self.logger.info(f"🔢 Total chunks: {total_chunks}")
        self.logger.info(f"🧵 Parallel workers: {max_workers}")
        self.logger.info(f"🔑 Available tokens: {len(self.tokens)}")
        self.logger.info(f"🔄 Max retries per chunk: {self.max_retries}")
        self.logger.info("=" * 80)
        
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
        
        # Process chunks in parallel
        all_data = []
        successful_chunks = 0
        failed_chunks = 0
        retry_stats = {'total_retries': 0, 'successful_retries': 0}
        
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
                    
                    # Track retry statistics
                    if 'attempts' in result and result['attempts'] > 1:
                        retry_stats['total_retries'] += result['attempts'] - 1
                        if result['success']:
                            retry_stats['successful_retries'] += result['attempts'] - 1
                    
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
                progress = (completed / total_chunks) * 100
                
                with self.lock:
                    self.logger.info(f"📈 Progress: {completed}/{total_chunks} ({progress:.1f}%) | ✅ {successful_chunks} | ❌ {failed_chunks} | 🔄 {retry_stats['total_retries']} retries")
        
        # Combine all data
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            self.logger.info(f"\n" + "=" * 80)
            self.logger.info(f"🎉 DATA RETRIEVAL COMPLETE!")
            self.logger.info(f"=" * 80)
            self.logger.info(f"📊 Total records: {len(final_df):,}")
            self.logger.info(f"📅 Date range: {final_df['datetime'].min()} to {final_df['datetime'].max()}")
            self.logger.info(f"🏢 Unique stations: {final_df['station_id'].nunique()}")
            self.logger.info(f"✅ Successful chunks: {successful_chunks}/{total_chunks}")
            self.logger.info(f"❌ Failed chunks: {failed_chunks}/{total_chunks}")
            self.logger.info(f"🔄 Total retries attempted: {retry_stats['total_retries']}")
            self.logger.info(f"🎯 Successful retries: {retry_stats['successful_retries']}")
            
            if failed_chunks > 0:
                self.logger.warning(f"⚠️  WARNING: {failed_chunks} chunks failed permanently - you may have missing data!")
            
            return final_df
        else:
            self.logger.error(f"\n" + "=" * 80)
            self.logger.error(f"❌ NO DATA RETRIEVED")
            self.logger.error(f"=" * 80)
            self.logger.error(f"✅ Successful chunks: {successful_chunks}/{total_chunks}")
            self.logger.error(f"❌ Failed chunks: {failed_chunks}/{total_chunks}")
            self.logger.error(f"🔄 Total retries attempted: {retry_stats['total_retries']}")
            return pd.DataFrame()


def setup_logging(log_filename: str = None) -> logging.Logger:
    """
    Set up logging configuration for the script.
    
    Args:
        log_filename (str): Name of the log file. If None, uses timestamp-based name.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"synoptic_data_retrieval_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # Also log to console for immediate feedback
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Log file created: {log_path}")
    
    return logger


def main():
    """Main function to run the script."""
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python synoptic_data_logger.py <start_date> <end_date>")
        print("Example: python synoptic_data_logger.py 2020-01-01 2021-01-01")
        sys.exit(1)
    
    START_DATE = sys.argv[1]
    END_DATE = sys.argv[2]
    
    # Validate date format
    try:
        datetime.strptime(START_DATE, '%Y-%m-%d')
        datetime.strptime(END_DATE, '%Y-%m-%d')
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        sys.exit(1)
    
    # Set up logging
    log_filename = f"synoptic_data_{START_DATE}_to_{END_DATE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_filename)
    
    # Configuration - Multiple API Tokens for Parallel Processing
    API_TOKENS = [
        "a2386b75ecbc4c2784db1270695dde73",
        "ecd8cc8856884bcc8f02f374f8eb87fc",
        "97985ec5837949cf807aa36544a7ca57",
        "1445303bab134661bdae8e1155482ff0",
        "265e9a7f586d45219a63a7afbe256b33",
        "2eb808b92c9841adb31e2b2608e9afc3",
        "ee3af9f0107142f98764d24fbfff3348",
        "31eaaf2ef5cc40519af576ba2bd0db65",
        "b5ee3feab5fc44f3a0a7ec254faf3ebe",
        "fbebef61121547fa9b1f1a066073b693"
    ]
    
    # Example CWAs (modify as needed)
    CWA_LIST = ["BYZ", "BOI", "LKN", "EKA", "FGZ", "GGW", "TFX", "VEF", "LOX", "MFR",
                "MSO", "PDT", "PSR", "PIH", "PQR", "REV", "STO", "SLC", "SGX", "MTR",
                "HNX", "SEW", "OTX", "TWC"]
    
    # Time chunk size (24 for daily, 12 for 12-hour chunks, 6 for 6-hour chunks)
    HOUR_CHUNK = 24
    
    # Retry configuration
    MAX_RETRIES = 3      # Maximum retry attempts per chunk
    RETRY_DELAY = 2.0    # Base delay between retries (with exponential backoff)
    
    # Number of parallel workers (None to use all available tokens)
    MAX_WORKERS = None
    
    # Log script start
    logger.info(f"🚀 Starting Synoptic Data Retrieval Script")
    logger.info(f"📅 Date range: {START_DATE} to {END_DATE}")
    logger.info(f"🔐 Process ID: {os.getpid()}")
    logger.info(f"📂 Working directory: {os.getcwd()}")
    
    try:
        # Initialize API client with multiple tokens and retry configuration
        api_client = SynopticDataAPIParallel(
            API_TOKENS, 
            max_retries=MAX_RETRIES, 
            retry_delay=RETRY_DELAY, 
            logger=logger
        )
        
        # Retrieve data with timing
        start_time = time.time()
        weather_data = api_client.get_data_range(
            cwas=CWA_LIST,
            start_date=START_DATE,
            end_date=END_DATE,
            hour_chunk=HOUR_CHUNK,
            max_workers=MAX_WORKERS
        )
        end_time = time.time()
        
        # Display results
        if not weather_data.empty:
            # Performance metrics
            total_time = end_time - start_time
            logger.info(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
            logger.info(f"📊 Records per second: {len(weather_data)/total_time:.0f}")
            
            # Set index
            weather_data.set_index(['datetime', 'station_id'], inplace=True)
            weather_data.sort_index(inplace=True)
            
            # Save to CSV
            output_filename = f"synoptic_data_WR1_{START_DATE}_to_{END_DATE}.csv"
            weather_data.to_csv(output_filename)
            logger.info(f"\n💾 Data saved to: {output_filename}")
            
            # Display sample of the data
            logger.info(f"\n🔍 Sample data (first 5 rows):")
            logger.info(f"\n{weather_data.head().to_string()}")
            
            # Final summary
            logger.info(f"\n🎉 SCRIPT COMPLETED SUCCESSFULLY!")
            logger.info(f"📊 Final dataset: {len(weather_data):,} records")
            logger.info(f"📁 Output file: {output_filename}")
            
        else:
            logger.error("❌ No data was retrieved.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 SCRIPT FAILED: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)
    
    finally:
        logger.info(f"📝 Full log available at: logs/{log_filename}")
        logger.info(f"🔚 Script execution ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()