import requests
import pandas as pd
import zipfile
import io
import os
from datetime import datetime, timedelta
from pathlib import Path
import time

class BinanceDataDownloader:
    def __init__(self, output_dir="binance_data"):
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Colunas padrão dos dados de klines da Binance
        self.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
    
    def generate_months(self, years_back=1):
        """Gera lista de meses para download baseado em anos anteriores"""
        # Data atual
        now = datetime.now()
        # Mês anterior (dados geralmente ficam disponíveis com 1 mês de atraso)
        end_date = now.replace(day=1) - timedelta(days=1)  # Último dia do mês anterior
        end_date = end_date.replace(day=1)  # Primeiro dia do mês anterior
        
        # Data de início
        start_date = end_date - timedelta(days=365 * years_back)
        start_date = start_date.replace(day=1)
        
        months = []
        current = start_date
        
        while current <= end_date:
            months.append(current.strftime("%Y-%m"))
            # Próximo mês
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
                
        return months
    
    def download_file(self, url, symbol, month):
        """Download de um arquivo específico"""
        try:
            print(f"Baixando {symbol} {month}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extrair dados do arquivo ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # Assumindo que há apenas um arquivo CSV no ZIP
                csv_filename = zip_file.namelist()[0]
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file, header=None, names=self.columns)
            
            # Converter timestamps para datetime (detectar formato automaticamente)
            try:
                # Detectar se é milissegundos ou microsegundos baseado no tamanho
                sample_timestamp = df['open_time'].iloc[0]
                
                if len(str(sample_timestamp)) > 13:  # Microsegundos (16 dígitos)
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='us', errors='coerce')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='us', errors='coerce')
                    print(f"  📅 Formato: microsegundos (2025+)")
                else:  # Milissegundos (13 dígitos ou menos)
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', errors='coerce')
                    print(f"  📅 Formato: milissegundos (2024-)")
                
                # Remover linhas com timestamps inválidos
                df = df.dropna(subset=['open_time', 'close_time'])
                
                # Verificar se há timestamps válidos
                if df.empty:
                    print(f"⚠️ Timestamps inválidos para {symbol} {month}")
                    return None
                    
            except Exception as e:
                print(f"⚠️ Erro na conversão de timestamps para {symbol} {month}: {e}")
                return None
            
            # Converter colunas numéricas
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'number_of_trades',
                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remover coluna 'ignore'
            df = df.drop('ignore', axis=1)
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Erro ao baixar {symbol} {month}: {e}")
            return None
        except Exception as e:
            print(f"Erro ao processar {symbol} {month}: {e}")
            return None
    
    def download_symbol_data(self, symbol, years_back=1):
        """Download de todos os dados de um símbolo"""
        months = self.generate_months(years_back)
        all_data = []
        
        symbol_dir = self.output_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        for month in months:
            url = f"{self.base_url}/{symbol}/1m/{symbol}-1m-{month}.zip"
            
            df = self.download_file(url, symbol, month)
            if df is not None and not df.empty:
                all_data.append(df)
                
            # Pausa para evitar sobrecarga do servidor
            time.sleep(0.5)
        
        if all_data:
            # Combinar todos os dados
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
            
            # Salvar em parquet
            output_file = symbol_dir / f"{symbol}_1m_{years_back}years.parquet"
            combined_df.to_parquet(output_file, index=False)
            
            print(f"✅ {symbol}: {len(combined_df):,} registros salvos em {output_file}")
            return combined_df
        else:
            print(f"❌ Nenhum dado encontrado para {symbol}")
            return None
    
    def download_all_symbols(self, symbols=None, years_back=1):
        """Download de dados para múltiplos símbolos"""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ETHBTC']
        
        print(f"Iniciando download para {len(symbols)} símbolos...")
        print(f"Período: {years_back} ano(s)")
        print(f"Intervalo: 1 minuto")
        print(f"Diretório de saída: {self.output_dir}")
        
        # Mostrar período de datas que será baixado
        months = self.generate_months(years_back)
        if months:
            print(f"Período dos dados: {months[0]} até {months[-1]}")
        print("-" * 50)
        
        results = {}
        
        for symbol in symbols:
            print(f"\n📊 Processando {symbol}...")
            df = self.download_symbol_data(symbol, years_back)
            results[symbol] = df
            
        print("\n" + "=" * 50)
        print("RESUMO:")
        for symbol, df in results.items():
            if df is not None:
                start_date = df['open_time'].min().strftime('%Y-%m-%d')
                end_date = df['open_time'].max().strftime('%Y-%m-%d')
                print(f"{symbol}: {len(df):,} registros ({start_date} até {end_date})")
            else:
                print(f"{symbol}: Falha no download")
        
        return results

# Exemplo de uso
if __name__ == "__main__":
    # PARÂMETROS CONFIGURÁVEIS
    YEARS_BACK = 2  # Altere aqui a quantidade de anos
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ETHBTC']  # Símbolos para download
    OUTPUT_DIR = "binance_data"  # Diretório de saída
    
    # Criar downloader
    downloader = BinanceDataDownloader(output_dir=OUTPUT_DIR)
    
    # Executar downloads
    results = downloader.download_all_symbols(
        symbols=SYMBOLS, 
        years_back=YEARS_BACK
    )
    
    print(f"\n🎉 Download concluído! Arquivos salvos em '{OUTPUT_DIR}/'")