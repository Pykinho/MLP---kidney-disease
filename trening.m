function[w1,w2,funkcja_kosztu]=trening(l_neuronow_wejsc,l_neuronow_ukrytych,l_neuronow_wyjsc,docelowy_blad,l_iteracji,eta,X,Ocz)

% funkcja trenująca sieć
% parametry wejściowe funkcji:
% l_neuronow_wejsc - liczba neuronów wejściowych
% l_neuronow_ukrytych - liczba neuronów ukrytych
% l_neuronow_wyjsc - liczba neuronow wyjsciowych
% docelowy_blad - błąd który nas zadowoli i po którego osiągnięciu sieć zakończy proces uczenia
% l_iteracji -  liczba iteracji, która zostanie wykonana w celu znalezienia docelowego błędu
% eta - współczynnik szybkości uczenia
% X - macierz z danymi wejsciowymi o wymiarze  l_neuronow_wejsc x N (liczba danych uczących sieć)
% Ocz - macierz z oczekiwanymi wyjściami o wymiarze l_neuronow_wyjs x N

% parametry wyjściowe funkcji
% 
% w1 - macierz wag warstwy ukrytej o wymiarze l_neuronow_ukrytych x l_neuronow_wejsc
% w2 - macierz wag warstwy wyjściowej o wymiarze l_neuronow_wyjsciowych x l_neuronow_ukrytych 
% funkcja kosztu - wektor zawierający wartości błędów, które następnie będziemy rysować na wykresie 

N = length(X); %liczba danych uczących sieć

w1 = rand(l_neuronow_ukrytych,l_neuronow_wejsc); % przyjęcie losowych wartości wag
w2 = rand(l_neuronow_wyjsc,l_neuronow_ukrytych); % przyjęcie losowych wartości wag

DW2 = zeros(l_neuronow_wyjsc,l_neuronow_ukrytych); %zmiana wprowadzana na macierzy wag warstwy wyjściowej używana w algorytmie wstecznej propagacji błędów
DW1 = zeros(l_neuronow_ukrytych,l_neuronow_wejsc); %zmiana wprowadzana na macierzy wag warstwy ukrytej używana w algorytmie wstecznej propagacji błędów
koszt_zbior = zeros(1,l_iteracji); % będzie używane do przechowywania wartości błędów zanim zostanie przekazane do wektora funkcja_kosztu


for i=1:l_iteracji
    
los_wektor = randperm(N); % losowanie kolejności podawania wektorów danych na wejście sieci w aktualnej iteracji
X = X(:,los_wektor); % modyfikacja macierzy podanej na wejście sieci - wymieszanie wektorów danych
Ocz = Ocz(:,los_wektor); % modyfikacja macierzy wartości oczekiwanych, aby po zmianie macierzy X być w stanie poprawnie wyliczyć funkcję kosztu

V = w1 * X; % obliczanie pobudzeń neuronów warstwy ukrytej
Z = 1 ./ (1+exp(-V)); %obliczanie stanu wyjść neuronów warstwy ukrytej po przepuszczeniu przez funkcję sigmoidalną

U = w2 * Z; % obliczanie stanu wyjść neuronów warstwy wyjściowej
Y = 1 ./ (1+exp(-U)); %Obliczanie stanu wyjść neuronów warstwy wyjściowej

E = Ocz - Y;
koszt = mean(mean(E.^2))/2; %Obliczanie wartości funkcji kosztu dla obecnej iteracji
dCdAl = E; %pochodna kosztu po aktywacji
koszt_zbior(i) = koszt; %dodajemy wartość funkcji kosztu dla obecnej iteracji do naszego zbioru

if (koszt<docelowy_blad)
    funkcja_kosztu = koszt_zbior(1:i);
    return %jeśli wartość błędu będzie mniejsza od tej pożądanej, to sieć osiągnęła swój cel i możemy opuścić funkcję
end
 

df1 = Y.*(1-Y); %pochodna funkcji sigmoidalnej, która będzie używana do obliczenia sygnału błędu warstwy wyjściowej
delta2 = dCdAl.*df1; % obliczenie sygnału błędu warstwy wyjściowej
DW2 = eta * delta2*Z'; %różnica o jaka zmodyfikowane zostaną wagi wyjściowe

df2= Z.*(1-Z); %pochodna funkcji sigmoidalnej, która będzie używana do obliczenia sygnału błędu warstwy ukrytej
delta1 = df2 .* (w2' * delta2); % obliczenie sygnału błędu warstwy ukrytej
DW1 = eta * delta1*X'; %różnica o jaką zmodyfikowane zostaną wagi ukryte

w2 = w2 + DW2; %aktualizacja wag warstwy wyjściowej
w1 = w1 + DW1; %aktualizacja wag warstwy ukrytej
end
funkcja_kosztu = koszt_zbior;
end