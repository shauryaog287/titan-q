import os
import time
import pandas as pd
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

class QuantDataLake:
    """
    High-performance local tick database using Parquet.
    Bypasses the need for a heavy SQL server by storing compressed columnar data.
    """
    def __init__(self, root_dir: str = "G:/My Drive/NSE_Data_Lake"):
        # Explicitly routed to Google Drive (G:). Bypasses C: and D: drives entirely.
        self.root_dir = root_dir
        self.tick_dir = os.path.join(self.root_dir, "ticks")
        os.makedirs(self.tick_dir, exist_ok=True)
        
        self._tick_buffer = []
        self._flush_size = 10000  # Flush to cloud disk every 10,000 ticks
        
    def insert_tick(self, symbol: str, price: float, volume: float, is_buyer_maker: bool):
        """Insert a single high-frequency tick into the memory buffer."""
        self._tick_buffer.append({
            "timestamp": time.time(),
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "is_buyer_maker": is_buyer_maker
        })
        
        if len(self._tick_buffer) >= self._flush_size:
            self.flush()
            
    def flush(self):
        """Write the memory buffer to a compressed Parquet file on SSD."""
        if not self._tick_buffer:
            return
            
        df = pd.DataFrame(self._tick_buffer)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Partition by date to keep file sizes optimal
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(self.tick_dir, f"ticks_{date_str}_{int(time.time())}.parquet")
        
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, compression='snappy')
        
        # Clear buffer
        self._tick_buffer = []
        
    def load_historical_ticks(self, symbol: str, start_date: str) -> pd.DataFrame:
        """Load massive amounts of tick data in milliseconds for the Hawkes / HMM models."""
        # For a full implementation, this would read multiple parquet files
        pass

if __name__ == "__main__":
    print("QuantDataLake initialized.")
    lake = QuantDataLake()
    lake.insert_tick("MOCK", 100.0, 1.5, True)
    lake.flush()
    print(f"Data lake directory created at: {lake.tick_dir}")
