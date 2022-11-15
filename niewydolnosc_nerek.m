clear all
close all

l_neuronow_wejsc = 20; %liczba neuronów wejściowych
l_neuronow_ukrytych = 5; %liczba neuronów ukrytych
l_neuronow_wyjsc = 1; %liczba neuronow wyjsciowych
docelowy_blad = 1e-10; %błąd który nas zadowoli i zakończy proces uczenia
l_iteracji = 1000000; % liczba iteracji, która zostanie wykonana w celu znalezienia docelowego błędu
eta = 0.02; %współczynnik szybkości uczenia


opts = detectImportOptions('Dane.xlsx');

opts.Sheet = 'dane znormalizowane_wynik';
opts.SelectedVariableNames = [24:43];
opts.DataRange = '2:281';
X = readmatrix('Dane.xlsx',opts)'; %wczytanie danych uczących do macierzy X z pliku Dane

opts.SelectedVariableNames = [22:22];
Ocz = readmatrix('Dane.xlsx',opts)'; %wczytanie oczekiwanych wartosci wyjsciowych dla danych uczących

tic;
[w1,w2,funkcja_kosztu]=trening(l_neuronow_wejsc,l_neuronow_ukrytych,l_neuronow_wyjsc,docelowy_blad,l_iteracji,eta,X,Ocz); %wywołanie funkcji uczącej sieć
toc;


opts.SelectedVariableNames = [24:43];
opts.DataRange = '282:401';
X = readmatrix('Dane.xlsx',opts)'; %wczytanie danych testowych do macierzy X z pliku Dane

opts.SelectedVariableNames = [22:22];
Ocz = readmatrix('Dane.xlsx',opts)';%wczytanie oczekiwanych wartosci wyjsciowych dla danych testowych

N = length (X); %liczba danych tetujących sieć

V = w1*X; % obliczanie pobudzeń neuronów warstwy ukrytej
Z = 1./(1+exp(-V)); %obliczanie stanu wyjść neuronów warstwy ukrytej po przepuszczeniu przez funkcję sigmoidalną
U = w2*Z; % obliczanie stanu wyjść neuronów warstwy wyjściowej

Y = 1./(1+exp(-U)); %Obliczanie stanu wyjść neuronów warstwy wyjściowej - diagnozy dotyczące niewydolności nerek przewidziane przez sieć

zgodne = 0;
czulosc = 0;
specyficznosc = 0;

for i=1:length(Ocz)
    
    if abs(Y(i)-Ocz(i))<1e-1
        zgodne = zgodne + 1;
    end
end

for i=1:75
    if abs(Y(i)-Ocz(i))<1e-1
        czulosc = czulosc + 1;
    end
end

for i=76:length(Ocz)
    if abs(Y(i)-Ocz(i))<1e-1
        specyficznosc = specyficznosc + 1;
    end
end

procentowa_dokladnosc = zgodne/length(Ocz) * 100 %procentowa skuteczność sieci na danych testowych
procentowa_czulosc = czulosc/75 * 100
procentowa_specyficznosc = specyficznosc/45 * 100

figure()
semilogy(funkcja_kosztu); 
title('Przebieg błędu w zależności od liczby iteracji')
xlabel('Liczba iteracji')
ylabel('Wartość błędu')
