# L2 – Sieci neuronowe: przewidywanie wyników egzaminu
 
## Opis zadania
 
Celem zadania jest wytrenowanie sieci neuronowej (regresja) w PyTorchu, która przewiduje wynik studenta z egzaminu na podstawie cech, takich jak liczba godzin nauki, ilość snu i inne czynniki opisane w zbiorze danych.
 
## Zakres realizacji
 
1. **Przygotowanie danych**
   - wczytanie zbioru z pliku CSV (`pandas`) i podział na cechy (`X`) oraz etykietę (`y`),
   - zmiana kształtu `y` z `(N,)` na `(N, 1)`, aby pasował do wyjścia modelu,
   - podział na zbiór treningowy (70%), walidacyjny (15%) i testowy (15%) za pomocą `train_test_split`,
   - konwersja tablic NumPy na tensory PyTorcha (`torch.tensor`).
2. **Model**
   - sieć zbudowana w `nn.Sequential` z kilku warstw gęstych (`nn.Linear`) z aktywacją `nn.ReLU`,
   - liczba wejść pierwszej warstwy równa liczbie cech,
   - warstwa wyjściowa z 1 neuronem i liniową aktywacją, bo model przewiduje jedną wartość liczbową.
3. **Trening**
   - optymalizator `optim.Adam` i funkcja straty `nn.MSELoss`,
   - pętla ucząca: `model.train()` → `optimizer.zero_grad()` → predykcja → strata → `loss.backward()` → `optimizer.step()`.
4. **Ewaluacja**
   - predykcje w trybie `model.eval()` z `torch.no_grad()`,
   - miara jakości: **MAE** (średni błąd bezwzględny),
   - eksperymenty z liczbą warstw, neuronów i funkcjami aktywacji w celu obniżenia MAE na zbiorze walidacyjnym,
   - zapamiętywanie wag modelu z najniższym MAE walidacyjnym (`copy.deepcopy(model.state_dict())`) i wczytanie ich (`load_state_dict`) przed testem końcowym,
   - jednorazowa ocena na zbiorze testowym i porównanie z wynikiem walidacyjnym.
## Technologie
 
Python, pandas, NumPy, scikit-learn, PyTorch

## Dane
**Etykiety**
   - Scores - procentowy wynik z egzaminu.
**Cechy**
   - Previous Scores – średnia wyników procentowych ze wszystkich poprzednich egzaminów na studiach.
   - Extracurricular Activities – czy dany student bierze udział w aktywnościach ponadprogramowych (np. maluje, uprawia sport).
   - Hours Studied – ile godzin ktoś uczył się do egzaminu.
   - Sleep Hours – ile godzin dziennie ktoś śpi.
   - Sample Question Papers Practiced – ile próbnych testów ktoś rozwiązał.
