# Pac-Man AI

**Pac-Man AI** è un progetto volto a creare un'Intelligenza Artificiale in grado di giocare a Pac-Man utilizzando l'algoritmo NEAT (NeuroEvolution of Augmenting Topologies). Il progetto è stato sviluppato come parte dell'esame di **Fondamenti di Intelligenza Artificiale** dell'**Università degli Studi di Salerno** nell'anno accademico 2024/2025.

## Caratteristiche Principali

### Rete Neurale Evolutiva (NEAT)

- Utilizzo di `neat-python` per gestire l'evoluzione dei pesi e della topologia della rete neurale.
- Possibilità di configurare in modo personalizzato i parametri di training (popolazione, probabilità di mutazione, soglia di compatibilità, ecc.).

### Ambiente di Gioco Basato su Nodi

- La mappa di Pac-Man viene trattata come un grafo con nodi e corridoi.
- Sistema di Breadth-First Search (BFS) per calcolare la distanza ai pellet più vicini o ai fantasmi, utile come informazione di input per la rete neurale.

### Strategie di Reward e Penalità

- **Ricompense** per mangiare pellet, power pellet e fantasmi in stato di vulnerabilità.
- **Penalità** per collisioni con fantasmi o se Pac-Man rimane troppo a lungo senza mangiare nulla (idle penalty).
- **Supporto al training stepwise**: una prima fase semplificata senza fantasmi, seguita da una fase finale con il gioco completo.

### Modalità di Esecuzione

- **Visuale**: Avvia il gioco con interfaccia grafica (`pygame`) e mostra in tempo reale l’apprendimento, con gli input dell'IA stampati in console (utile per il debug, ma più lento).
- **Headless**: Esegue il training senza rendering grafico, molto più veloce (sequenziale o parallelizzato).

## Struttura Principale dei File

### `run.py`

File principale. Contiene:

- Le classi `GameController` e `GameControllerStep`, che gestiscono la logica di Pac-Man (movimenti, collisioni, punteggio, ecc.).
- Le funzioni di valutazione per i genomi NEAT (`eval_genomes_visual`, `eval_genomes_headless`, ecc.).
- Le modalità di allenamento (classico, stepwise, visuale, headless, parallelizzato).
- La funzione `main` (blocco `if __name__ == "__main__":`) che mostra un menù per scegliere come avviare il gioco o il training.

### `visualize.py`

File di utilità (fornito da `neat-python` e adattato) per tracciare le statistiche della popolazione, disegnare la topologia delle reti, ecc.

### Altri File

- `constants.py`, `nodes.py`, `pellets.py`, `ghosts.py`, `pacman.py`, ecc.
- Insieme di moduli che gestiscono costanti, entità di gioco e meccaniche.

### `neat-config.txt`

File di configurazione di NEAT, dove si definiscono i parametri di evoluzione (popolazione, tassi di mutazione, ecc.).

## Requisiti

- Python 3.11
- `pygame`
- `neat-python`
- `numpy`
- `graphviz`
- `matplotlib`

### Installazione delle Dipendenze

Eseguire il seguente comando nel terminale:

```bash
pip install pygame neat-python numpy graphviz matplotlib
```

## Modalità di Avvio

Avviare lo script principale `run.py`. Da terminale:

```bash
python run.py
```
## Avvio di un Training Rapido
Verificare la presenza del file neat-config.txt (o rinominalo se necessario) e una volta avviato lo script principale,

Scegliere **1) Allenare in modalità classica NEAT** e poi una delle sottovoci:

a) Visuale
b) Headless sequenziale
c) Headless parallelizzata

Attendere il completamento delle generazioni impostate nel codice (valore di default: 150).
Il miglior genoma verrà salvato come **winner.pkl**.

## Crediti
Codice del gioco base Pac-Man: **https://pacmancode.com/** (modificato per adattarlo al training con NEAT).

NEAT: **https://neat-python.readthedocs.io/en/latest/**
