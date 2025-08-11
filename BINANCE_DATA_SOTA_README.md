# Binance Data Downloader SOTA - Documentação

## Visão Geral

O `binanceDataloader.py` e a célula correspondente no notebook foram otimizados especificamente para trabalhar com **dados de 1 minuto (m1)** seguindo padrões de estado da arte para machine learning em crypto.

## Características SOTA Implementadas

### 1. Dados Exclusivamente de 1 Minuto
- ✅ Resolução temporal máxima para captura de micro-padrões
- ✅ Ideal para estratégias de trading de alta frequência 
- ✅ Padrão da indústria para modelos modernos de crypto

### 2. Estrutura de Paths Otimizada
```
binance_data/
├── BTCUSDT/
│   └── BTCUSDT_1m_0.5years.parquet
├── ETHUSDT/
│   └── ETHUSDT_1m_0.5years.parquet
└── ...
```
- ✅ Compatível com `CryptoDatasetBuilder`
- ✅ Organização por ativo para eficiência
- ✅ Nomenclatura consistente e descritiva

### 3. Robustez e Validação
- ✅ Detecção automática de formato timestamp (ms/μs)
- ✅ Validação de dados baixados
- ✅ Tratamento de erros de rede
- ✅ Fallback para dados sintéticos no notebook

### 4. Configurações SOTA para o Notebook

#### Parâmetros Temporais
- **Context Length**: 1440 minutos (24 horas)
- **Prediction Length**: 60 minutos (1 hora)
- **Batch Size**: 16 (otimizado para dados de alta frequência)

#### Features SOTA
- **Dataset Unificado**: Todos os ativos em um dataset
- **Treinamento Anônimo**: Remove identificadores durante treino
- **Normalização por Janela**: Foca na forma dos padrões
- **Features Cíclicas**: Componentes temporais (sin/cos)

## Melhorias Implementadas

### No `binanceDataloader.py`:
1. **Removida interface CLI desnecessária** - O notebook usa a classe diretamente
2. **Foco exclusivo em dados de 1 minuto** - Não é parametrizável pois é o padrão SOTA
3. **Documentação melhorada** - Explicação clara do propósito e uso
4. **Código simplificado** - Mantém apenas funcionalidades essenciais

### No Notebook:
1. **Célula de download robusta** - Inclui validação de paths e fallback
2. **Configuração SOTA** - Parâmetros otimizados para dados de 1 minuto
3. **Análise exploratória específica** - Visualizações focadas em alta frequência
4. **Validação completa** - Verificação de estrutura e qualidade dos dados

## Uso no Kaggle

O notebook está otimizado para execução no Kaggle com:
- ✅ Paths absolutos corretos
- ✅ Fallback para dados sintéticos se necessário
- ✅ Configurações de memória otimizadas
- ✅ Batch sizes adequados para recursos limitados

## Estado da Arte - Por que 1 Minuto?

1. **Resolução Temporal Máxima**: Captura todos os movimentos significativos
2. **Padrões Intra-hora**: Essencial para trading moderno
3. **Volume de Dados**: Suficiente para treinamento robusto
4. **Padrão da Indústria**: Usado pelos melhores sistemas de trading
5. **Granularidade Ótima**: Equilibra ruído vs. informação útil

## Próximos Passos

Para melhorar ainda mais o pipeline:
1. Adicionar mais ativos crypto
2. Implementar features técnicas avançadas
3. Otimizar ainda mais os parâmetros do modelo
4. Adicionar validação cruzada temporal
