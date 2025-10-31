# Clash Royale Facial Emoji Recognition

Este projeto utiliza **Mediapipe**, **OpenCV** e **scikit-learn** para detectar expressões faciais através da webcam e exibir emotes do jogo Clash Royale conforme a emoção identificada.

---

## Funcionalidade

1. **Coleta de dados**: Captura imagens da webcam para diferentes expressões faciais
2. **Treinamento do modelo**: Cria um classificador de machine learning com os dados coletados
3. **Reconhecimento em tempo real**: Detecta expressões faciais e exibe emotes correspondentes

---

## Requisitos

- Python 3.10
- Windows / macOS / Linux
- Webcam funcionando

### Bibliotecas necessárias:
```bash
mediapipe
opencv-python
scikit-learn
joblib
numpy
pillow
playsound
```

---

## Configurando o ambiente

1. **Abra o terminal** no diretório do projeto

2. **Crie o ambiente virtual**:
```powershell
py -3.10 -m venv venv
```

3. **Ative o ambiente virtual**:
```powershell
venv\Scripts\Activate.ps1
```

4. **Permita a execução de scripts PowerShell** (apenas na primeira vez):
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

5. **Instale as dependências**:
```powershell
pip install mediapipe opencv-python scikit-learn joblib numpy pillow playsound
```

---

## Executando o Projeto (Fluxo Completo)

### Passo 1: Coleta de Dados

Execute o script para coletar imagens de treinamento:

```powershell
python collectData.py
```

**Como funciona:**
- A webcam será aberta mostrando a tela "Collect"
- Para cada expressão (crying, thumbsUp, angry, laughing, neutral), pressione **ENTER** para iniciar a coleta
- O script capturará automaticamente **300 amostras** por expressão
- Pontos faciais serão mostrados em tempo real na tela
- As features faciais serão salvas como arquivos `.npy` na pasta `data/`
- Pressione **ESC** para sair a qualquer momento

### Passo 2: Treinamento do Modelo

Após coletar os dados, treine o classificador:

```powershell
python trainModel.py
```

**O que acontece:**
- Carrega todos os dados `.npy` da pasta `data/`
- Extrai características faciais normalizadas
- Treina um modelo **SVM com kernel RBF** (Support Vector Machine)
- Divide os dados em treino (85%) e teste (15%)
- Mostra o **relatório de classificação** com métricas de acurácia
- Salva o modelo treinado como `models/face_expr_model.joblib`

### Passo 3: Reconhecimento em Tempo Real

Com o modelo treinado, execute o reconhecimento:

```powershell
python realtime.py
```

**Funcionalidades:**
- Detecta rostos em tempo real pela webcam
- Classifica a expressão facial usando o modelo treinado
- Exibe o **emote visual** do Clash Royale no canto inferior direito
- Toca o **som correspondente** à expressão detectada
- Mostra a **label e confiança** da predição no topo da tela
- **Threshold de confiança**: 10% (ajustável via `THRESHOLD`)
- **Delay entre sons**: 2 segundos (evita repetição excessiva)

---

## Estrutura do Projeto

```
clash-royale/
│
├── collectData.py          # Coleta features faciais para treinamento
├── trainModel.py           # Treina o modelo de classificação
├── realtime.py             # Reconhecimento em tempo real + emotes
├── utils.py                # Funções auxiliares (extração de features)
├── models/                 # Pasta para modelos treinados
│   └── face_expr_model.joblib  # Modelo salvo (gerado após treinamento)
├── data/                   # Pasta com os dados coletados
│   ├── thumbsUp/           # Features para joinha
│   ├── laughing/           # Features para risada
│   ├── crying/             # Features para choro
│   ├── angry/              # Features para raiva
│   └── neutral/            # Features para neutro
├── emotes/                 # Imagens PNG dos emotes do Clash Royale
│   ├── thumbsUp.png
│   ├── laughing.png
│   ├── crying.png
│   └── angry.mp3
├── sounds/                 # Sons MP3 dos emotes do Clash Royale
│   ├── thumbsUp.mp3
│   ├── laughing.mp3
│   ├── crying.mp3
│   └── angry.mp3
├── venv/                   # Ambiente virtual
└── README.md
```

---

## Treinamento Personalizado

Se quiser melhorar o modelo:

1. **Colete mais dados** com `collectData.py`
2. **Ajuste parâmetros** em `trainModel.py`:
```python
# Exemplo de ajuste de parâmetros do SVM
clf = SVC(
    kernel="rbf", 
    probability=True, 
    class_weight="balanced", 
    random_state=42,
    C=1.0,           # Parâmetro de regularização
    gamma='scale'     # Coeficiente do kernel
)
```

3. **Execute novamente** o fluxo completo:
```powershell
python collectData.py
python trainModel.py
python realtime.py
```

---

## Dicas para Melhor Desempenho

- **Iluminação**: Mantenha o rosto bem iluminado e uniforme
- **Posição**: Fique centralizado na câmera a ~50cm de distância
- **Expressões**: Faça expressões exageradas durante a coleta de dados
- **Consistência**: Mantenha a mesma expressão durante toda a coleta de cada label
- **Background**: Use fundo neutro para melhor detecção

---

## Solução de Problemas

**Problema**: Webcam não abre  
**Solução**: Verifique se outra aplicação não está usando a câmera

**Problema**: Erro "ModuleNotFoundError"  
**Solução**: Ative o ambiente virtual e reinstale as dependências

**Problema**: Baixa acurácia nas predições  
**Solução**: 
- Colete mais dados de treinamento (aumente `SAMPLES_PER_LABEL`)
- Verifique a iluminação e posição do rosto
- Ajuste o `THRESHOLD` no `realtime.py`

**Problema**: Sons não tocam  
**Solução**: Verifique se os arquivos MP3 estão na pasta `sounds/` com nomes corretos

**Problema**: Emotes não aparecem  
**Solução**: Verifique se as imagens PNG estão na pasta `emotes/` com nomes corretos

*Nota: Execute os scripts na ordem indicada para garantir o funcionamento correto do sistema.*

---
