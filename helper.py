import matplotlib.pyplot as plt

plt.ion()

# Создаем фигуру и оси один раз
fig, ax = plt.subplots(figsize=(10, 6))

def plot(scores, mean_scores):
    ax.cla()  # Очищаем текущие оси
    ax.set_title('Training...')
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Score')
    ax.plot(scores, label='Score')
    ax.plot(mean_scores, label='Mean Score')
    ax.set_ylim(ymin=0)
    ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    ax.legend(loc='upper left')
    plt.draw()
    plt.pause(0.01)