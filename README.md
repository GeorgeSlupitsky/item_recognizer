# 🧠 Custom CNN Classifier for Collectibles

Цей проєкт — вебзастосунок для класифікації зображень колекційних предметів за допомогою згорткової нейромережі (CNN). Крім класифікації, він також витягує текст з фото книг або вінілів (OCR через Tesseract).

## 🧩 Класи
- `books`
- `coins`
- `drumsticks`
- `stamps`
- `vinyls`

## 🧱 Архітектура моделі

Модель створена вручну (custom CNN) без попередньо навчених моделей. Вона складається з таких шарів:

- 3 × `Conv2D` + `MaxPooling2D`
- `Flatten`
- `Dense(256)` + `Dropout(0.5)`
- `Dense(5, activation='softmax')`

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
```

## ⚙️ Характеристики тренування

- 📐 Розмір зображення: `224×224`
- 🔄 Аугментація: повороти, зсув, зум, фліп
- ⚖️ Балансування класів: `class_weight`
- 📊 Стратифікований train/test спліт: `80/20`
- 🧠 Optimizer: `Adam`
- ⏱ Epochs: `20`

## 📈 Результати

- **Final Accuracy:** `75%`
- **Precision (macro avg):** `0.76`
- **Recall (macro avg):** `0.74`
- **F1-score (macro avg):** `0.75`

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| books        | 0.86      | 0.77   | 0.81     |
| coins        | 0.78      | 0.81   | 0.79     |
| drumsticks   | 0.94      | 0.66   | 0.78     |
| stamps       | 0.70      | 0.78   | 0.74     |
| vinyls       | 0.53      | 0.71   | 0.61     |

## 📊 Графіки

### 📌 Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### 📌 Accuracy / Loss
![Accuracy / Loss](accuracy_loss.png)


## 🌐 Веб-інтерфейс

- Побудований на Flask
- Drag & Drop підтримка (JS + CSS)
- Виводить:
  - Категорію
  - Впевненість
  - Текст (для книг/вінілів)

![](./interface_example.png)

## 📦 Встановлення

```bash
poetry install
poetry shell
pip install tensorflow-macos tensorflow-metal
python app.py
```

## 🗂 Структура

```
.
├── app.py                          # Flask API для завантаження фото
├── utils.py                        # Функції: класифікація, OCR
├── best_custom_cnn.keras           # Збережена модель
├── README.md                       # Документація
├── pyproject.toml                  # Poetry конфігурація
├── poetry.lock                     # Залежності (генерується)
├── .gitignore                      # Ігнорує тимчасові/непотрібні файли
│
├── accuracy_loss.png               # Графік точності та втрат
├── confusion_matrix.png            # Матриця плутанини
│
├── static/                         # Статичні файли (JS/CSS)
│   ├── styles.css
│   └── script.js
│
├── templates/                      # HTML шаблони Flask
│   └── index.html
│
├── test/                           # Тестові зображення
│   ├── Coin.jpg
│   ├── Drumstick.jpg
│   ├── Stamp.jpg
│   └── Vinyl.jpg
│
├── train_model/                    # Навчання моделі
│   ├── train_model_cnn.py
│   └── dataset.zip                 # Dataset
│
└── image_collector/                # Збір зображень з інтернету
    ├── image_collector_ebay.py
    └── image_collector_google.py
```
