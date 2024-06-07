import math
from datetime import datetime


def calculate_biorhythms(
    day, physical_cycle=23, emotional_cycle=28, intellectual_cycle=33
):
    physical = math.sin((2 * math.pi * day) / physical_cycle)
    emotional = math.sin((2 * math.pi * day) / emotional_cycle)
    intellectual = math.sin((2 * math.pi * day) / intellectual_cycle)
    return physical, emotional, intellectual


def main():
    # Prośba o podanie danych użytkownika
    name = input("Podaj swoje imię: ")
    year = int(input("Podaj rok urodzenia (np. 2000): "))
    month = int(input("Podaj miesiąc urodzenia (np. 1 dla stycznia): "))
    day = int(input("Podaj dzień urodzenia: "))

    # Obliczenie daty urodzenia użytkownika w dniach
    birth_date = datetime(year, month, day)
    today = datetime.now()
    days_since_birth = (today - birth_date).days

    # Obliczenie biorytmów
    physical, emotional, intellectual = calculate_biorhythms(days_since_birth)

    # Sprawdzenie wyników biorytmów
    if physical > 0.5:
        print("Twoja fala fizyczna jest wysoka!")
    else:
        print("Twoja fala fizyczna jest niska.")

    if emotional > 0.5:
        print("Twoja fala emocjonalna jest wysoka!")
    else:
        print("Twoja fala emocjonalna jest niska.")

    if intellectual > 0.5:
        print("Twoja fala intelektualna jest wysoka!")
    else:
        print("Twoja fala intelektualna jest niska.")

    # Sprawdzenie prognozy na następny dzień
    next_day_physical, next_day_emotional, next_day_intellectual = calculate_biorhythms(
        days_since_birth + 1
    )
    if (
        next_day_physical > physical
        or next_day_emotional > emotional
        or next_day_intellectual > intellectual
    ):
        print("Nie martw się, jutro będzie lepiej!")


if __name__ == "__main__":
    main()

### uzyskanie poprawnego programu poprzez wygenerowanie calego kodu zajelo mi jakies 3min, program byl wygenerowany poprawnie poza jednym drobnym bledem, ktory poprawilem recznie
