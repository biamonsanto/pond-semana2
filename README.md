# Transformer – Experimento CPU vs GPU (T4) no Google Colab

Este repositório documenta um experimento de treinamento de um Transformer no **Google Colab** comparando **CPU** e **GPU (T4)**, com controle explícito de `steps_per_epoch` para medir tempo.

## Estrutura do projeto

- `tentativa-com-cpu/`
  - `transformer-2.ipynb`
- `tentativa-com-gpu/` (T4)
  - `transformer-2.ipynb`

## Objetivo

- Medir e comparar o tempo de treinamento entre CPU e GPU (T4) no Colab.
- Investigar por que, neste caso específico, a GPU **não foi mais rápida** e ainda **crashou** antes de concluir a época.

---

## Resultados (com `steps_per_epoch` limitado)

| Ambiente             | Dispositivo | `steps_per_epoch` | Steps concluídos | Tempo até evento | Estado final | Tempo/step (aprox.) |
|---------------------:|:-----------:|:-----------------:|:----------------:|:----------------:|:------------:|:-------------------:|
| tentativa-com-cpu    | CPU (Colab) | 200               | 200              | **41 min**       | Concluído    | **≈ 12,3 s**        |
| tentativa-com-gpu    | GPU T4      | 200               | **132**          | **~43 min**      | **Crash**    | **≈ 19,5 s**        |

**Nota:** Na corrida da **CPU** eu **mudei o código original** para rodar **200 passos em 1 época**, a fim de estimar o tempo total de uma época. Na **GPU (T4)**, com configuração equivalente, a execução **crashou** por volta dos **132 steps** (~43 min).

---

## Interpretação: por que a GPU (T4) não foi mais rápida e ainda crashou?

Aprendizados práticos desta experiência:

1. **Pipeline de dados CPU-bound / I/O-bound**  
   Mesmo usando GPU, o *input pipeline* (tokenização, `tf.data`, leitura do Drive) roda no **CPU** e pode virar gargalo.  
   - Sem `cache()` e `prefetch(tf.data.AUTOTUNE)`, a GPU fica ociosa esperando dados.  
   - Se a tokenização/preprocessamento roda a cada step, a vantagem da GPU some.

2. **Batch pequeno e/ou modelo leve ⇒ subutilização da GPU**  
   GPUs brilham com lotes maiores e sequências longas. Com lotes pequenos/curtos, o overhead de **transferência CPU→GPU** e de **lançamento de kernels** pode superar o ganho do paralelismo.

3. **Sem mixed precision (FP16) e sem XLA**  
   Rodar em **FP32** na T4 reduz throughput. Em Transformers, **`mixed_float16`** costuma dar salto de desempenho; XLA JIT pode ajudar adicionalmente (quando estável).

4. **Gargalos silenciosos no `tf.data`**  
   Sem `num_parallel_calls=tf.data.AUTOTUNE`, *map* e *batch* ficam sub-otimizados. Sem `cache()` (RAM) ou *snapshot*, a leitura do Drive deixa tudo mais lento/variável.

5. **Causas prováveis do crash (no Colab)**  
   - **OOM (VRAM)** por sequência longa, *batch* grande ou *dataloader* sem cuidado.  
   - **Instabilidade do runtime/limitação de sessão** (reinícios transitórios).  
   - **Incompatibilidades de versão** (TF/TF-Text/kernels) gerando falhas em steps aleatórios.

> **Conclusão didática:** GPU não é “mágica”. Sem cuidar do *input pipeline*, tamanho de lote, **mixed precision** e leitura de dados, a T4 pode ficar **mais lenta que a CPU** e, sob pressão de memória, **crashar**.

---

## Como reproduzir (resumo operacional)

1. **Habilitar GPU T4 no Colab**  
   *Runtime → Change runtime type → Hardware accelerator: GPU (T4)*

2. **Garantir versões compatíveis de TensorFlow e TensorFlow Text**  
   Instale `tensorflow-text` **na mesma versão** do `tensorflow`.

3. **Confirmar GPU**  
   `tf.config.list_physical_devices("GPU")` deve listar a GPU.

4. **Otimizações recomendadas (GPU)**  
   - **Mixed precision** (FP16): `mixed_precision.set_global_policy("mixed_float16")`  
   - **XLA JIT** (se estável na sua versão)  
   - `tf.data` com `cache()`, `map(..., num_parallel_calls=AUTOTUNE)` e `prefetch(AUTOTUNE)`  
   - **Aumentar `BATCH_SIZE`** até o limite da VRAM

---

## Onde limitei os passos por época

Para comparabilidade do experimento, limitei **steps por época** no `model.fit`:

```python
history = model.fit(
    train,
    epochs=1,    # <- limite modificado do tutorial 
    steps_per_epoch=200,        # <- limite modificado do tutorial
    validation_data=val,
    # validation_steps=...
)
 ```

## Avaliação do Código

### ✅ Pontos positivos
- **Encapsulamento de inferência** via classe `Translator(tf.Module)`: centraliza tokenização, preparação do input e laço de decodificação em um único ponto.
- **Defensivo na entrada** (`assert isinstance(sentence, tf.Tensor)` e correção de rank com `tf.newaxis`): reduz erros silenciosos de shape.
- **Uso correto dos tokenizers salvos** (`tf.saved_model.load(...)`): garante consistência entre treinamento e inferência sem recriar vocabulários.
- **Tratamento explícito de tokens especiais** (`start`/`end` obtidos do tokenizer): evita “mágicas” e facilita manutenção.
- **Pipeline Keras idiomático** (`model.fit(..., validation_data=...)`): aproveita callbacks, logs e histórico padrão.
- **Separação de responsabilidades** modelo ↔ tokenização ↔ laço de inferência: facilita debugar cada etapa.
- **Compatibilidade com grafos**: estrutura do `Translator.__call__` é naturalmente “traceable” para `@tf.function` (futuro ganho de performance).
- **Clareza de etapas** (tokenizar → encoder_input → laço de geração): leitura fácil para quem está aprendendo seq2seq/Transformer.


### ⚠️ Pontos negativos (oportunidades de melhoria)
- **Instrumentação insuficiente de tempo**: não há medição programática de *wall-clock* nem tempo médio por *batch* (ex.: `time.perf_counter()` e `Callback` de *batch timer*).
- **Diagnóstico de GPU ausente**: faltam checagens como `tf.config.list_physical_devices("GPU")` e `!nvidia-smi` (antes e durante o treino) para verificar VRAM/uso e investigar o crash.
- **Reprodutibilidade frágil de dependências**: versões de `tensorflow` e `tensorflow-text` não estão “pinadas” nem documentadas; risco de *mismatch* no Colab.
- **Pipeline de dados não documentado/otimizado**: não ficou evidenciado o uso de `cache()`, `prefetch(tf.data.AUTOTUNE)` e `num_parallel_calls=tf.data.AUTOTUNE`; sem isso, a GPU pode ficar ociosa.
- **Batch size subótimo para GPU**: não há varredura de `BATCH_SIZE` específica para a T4 (tentar 128/256 se couber) para aumentar *throughput*.
- **Sem mixed precision (FP16)**: ausência de `mixed_precision.set_global_policy("mixed_float16")` reduz fortemente o ganho na T4.
- **Tratamento de OOM/crash**: não há coleta de logs/stacktrace, nem *fallbacks* (reduzir `max_len`, `BATCH_SIZE`, limpar cache) ao detectar OOM.
- **Controle de aleatoriedade não fixado**: `seed`/`tf.random.set_seed` ausentes, dificultando comparações exatas entre execuções.
- **Validação possivelmente desbalanceada**: `validation_steps` não calibrado ao tamanho real do conjunto pode alongar validação sem necessidade.
- **Gestão de artefatos**: problema reportado com diretórios duplicados de *tokenizers* (SavedModel); é preciso consolidar a árvore e validar o caminho antes do `load`.


## Conclusão

O experimento mostrou de forma clara que **ter GPU não garante ganho de desempenho**: com `steps_per_epoch` padronizado, a **CPU concluiu 200/200 steps em ~41 min**, enquanto a **T4 travou aos 132/200 steps em ~43 min**. O resultado é consistente com **gargalo de input pipeline no CPU/I/O**, **batch size pequeno**, **ausência de `mixed_precision` (FP16)** e **possíveis limitações/instabilidades do runtime**. Como aprendizado, fica evidente que **configuração e otimização do fluxo de dados** (cache/prefetch/parallel calls), **escolhas de hiperparâmetros** (batch, max_len) e **compatibilidade de versões** (TF/TF-Text) são tão críticas quanto o próprio hardware.

**O que fazer para reverter o quadro (resumo prático):**
- Ativar **`mixed_precision`** e **aumentar `BATCH_SIZE`** na T4 (até o limite de VRAM).
- Otimizar `tf.data` com **`cache()`**, **`prefetch(AUTOTUNE)`** e **`num_parallel_calls=AUTO`**.
- **Fixar versões** de `tensorflow` e `tensorflow-text` (mesma versão), registrar `!nvidia-smi` e `tf.config.list_physical_devices("GPU")`.
- **Instrumentar tempo** (wall-clock e por batch) e **coletar logs** em caso de OOM/crash (reduzir `max_len`/`BATCH_SIZE` se necessário).

Em síntese: a experiência comprovou, na prática, que **desempenho em DL é sistêmico** — depende de **dados, software e hardware** trabalhando em conjunto. Com os ajustes propostos, a tendência é a GPU **superar a CPU** de forma consistente em próximos testes.
