import numpy as np
import datetime


def get_birthdate():
    name = input("Podaj imię: ")
    month = int(input("Podaj miesiąc urodzenia: "))
    day = int(input("Podaj dzień urodzenia: "))
    year = int(input("Podaj rok urodzenia: "))
    return name, month, day, year


def calculate_days_since_birth(month, day, year):
    birthdate = datetime.date(year, month, day)
    today = datetime.date.today()
    delta = today - birthdate
    return delta.days


def physical_wave(week):
    return np.sin(week * 2 * np.pi / 23)


def emotional_wave(week):
    return np.sin(week * 2 * np.pi / 28)


def intellectual_wave(week):
    return np.sin(week * 2 * np.pi / 33)


def print_waves_info(name, days):
    print(f"Witaj {name}!")
    print("Twój numer to:", days)
    fizyczna = physical_wave(days)
    emocjonalna = emotional_wave(days)
    intelektualna = intellectual_wave(days)
    print("Twoja fala fizyczna:", fizyczna)
    print("Twoja fala emocjonalna:", emocjonalna)
    print("Twoja fala intelektualna:", intelektualna)
    return fizyczna, emocjonalna, intelektualna


def analyze_wave(wave, name):
    if wave > 0.5:
        print(f"Dzisiaj jest dobry dzień na {name}, gratuluję!")
    elif wave < -0.5:
        if wave < wave_tomorrow:
            print(f"Jutro będzie lepszy {name}")
        else:
            print(f"Dzisiaj jest dobry dzień na odpoczynek od {name}")


name, month, day, year = get_birthdate()
days_since_birth = calculate_days_since_birth(month, day, year)
fizyczna, emocjonalna, intelektualna = print_waves_info(name, days_since_birth)

wave_tomorrow = physical_wave(days_since_birth + 1)
analyze_wave(fizyczna, "ćwiczenia")
analyze_wave(emocjonalna, "spotkanie z przyjaciółmi")
analyze_wave(intelektualna, "naukę")
