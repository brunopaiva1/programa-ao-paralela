import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def carregar_histograma(caminho_arquivo):
    labels = []
    contagens = []

    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            if linha.startswith("Bin"):
                partes = linha.strip().split(":")
                faixa = partes[0].split("(", 1)[1].split(")")[0] 
                count = int(partes[1])
                labels.append(faixa)
                contagens.append(count)

    return labels, contagens

def plotar_histograma(labels, contagens):
    plt.figure(figsize=(12, 6))
    plt.bar(labels, contagens, width=0.8, edgecolor='black')
    plt.xlabel("Intervalo do Bin")
    plt.ylabel("Frequência")
    plt.title("Histograma Global (Resultado MPI)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    caminho_arquivo = "/home/bruno/programacao-paralela/MPI/Histograma/histograma.txt"
    bins, contagens = carregar_histograma(caminho_arquivo)
    plotar_histograma(bins, contagens)
