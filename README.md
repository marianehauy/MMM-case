# Marketing Mix Modeling (MMM) Pipeline
Este repositório contém um projeto de Marketing Mix Modeling (MMM) desenvolvido para analisar o impacto dos investimentos de mídia em ganhos de tráfego. O modelo foi construído para auxiliar na tomada de decisões sobre alocação de orçamento e otimização de campanhas de marketing.

## Pipeline do Projeto
O pipeline do projeto é composto pelas seguintes etapas:

1. Agregação em Semanas: Os dados de mídia e tráfego são agregados em intervalos semanais para garantir consistência temporal e melhorar a robustez das análises.

2. Adição de Adstock nas Mídias: A transformação de adstock geométrico com delay é aplicada para modelar o efeito de carryover das campanhas de mídia ao longo do tempo.

3. Escalonamento e Normalização das Features: As variáveis de mídia e controle são escalonadas e normalizadas para trazer os dados para uma escala comparável, o que é crucial para a modelagem e interpretação.

4. Adição de Saturação nas Mídias: Uma transformação de saturação é aplicada para capturar o efeito decrescente de retornos marginais em altos níveis de investimento.

5. Regressão Usando Ridge com Constraints: A regressão Ridge é utilizada para modelar o impacto das variáveis explicativas no tráfego. Restrições são aplicadas para garantir que os coeficientes de mídia sejam positivos, refletindo que maiores investimentos não reduzem o tráfego.

6. Simulação de Investimento: Simulações são realizadas para estimar o impacto incremental de R$1 investido em cada canal de mídia sobre o ganho de tráfego, permitindo a avaliação do ROI das campanhas.

## Estrutura do Repositório
- data/: Contém os dados brutos e transformados.
- notebooks/: Notebooks com as etapas de análise exploratória e desenvolvimento do modelo.
- src/: Código fonte do pipeline de modelagem e funções auxiliares.
- models/: Modelos treinados e resultados das simulações.
- reports/: Relatórios e visualizações geradas durante o projeto.

## Como Executar
Clone o repositório:

1. git clone https://github.com/username/mmm-project.git
2. cd mmm-project
3. Instale as dependências:
- Ative o ambiente virtual executando `poetry shell`.
- Instale as dependências do projeto executando `poetry install`.
4. Rode python src/run_pipeline.py

## Resultados e Interpretação
Os resultados do modelo são apresentados como coeficientes que representam o impacto das variáveis sobre o tráfego. A simulação de investimento fornece insights sobre como diferentes alocações de orçamento podem otimizar o retorno sobre o investimento medido em tráfego.

## Contribuições
Contribuições são bem-vindas! Por favor, abra uma issue ou envie um pull request.

## Licença
Este projeto é licenciado sob a MIT License.


