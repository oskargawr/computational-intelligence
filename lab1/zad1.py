import numpy
import datetime

name = input("Podaj imię: ")

month = int(input("Podaj miesiąc urodzenia: "))
day = int(input("Podaj dzień urodzenia: "))
year = int(input("Podaj rok urodzenia: "))


def change_birthdate_to_days(month, day, year):
    birthdate = datetime.date(year, month, day)
    today = datetime.date.today()
    delta = today - birthdate
    return delta.days


def physical_wave(week):
    return numpy.sin(week * 2 * numpy.pi / 23)


def emotional_wave(week):
    return numpy.sin(week * 2 * numpy.pi / 28)


def intellectual_wave(week):
    return numpy.sin(week * 2 * numpy.pi / 33)


print("Witaj", name, "!")
print("Twój numer to:", change_birthdate_to_days(month, day, year))

days = change_birthdate_to_days(month, day, year)

# print("Twoja fala fizyczna:", physical_wave(days))
# print("Twoja fala emocjonalna:", emotional_wave(days))
# print("Twoja fala intelektualna:", intellectual_wave(days))

fizyczna = physical_wave(days)
emocjonalna = emotional_wave(days)
intelektualna = intellectual_wave(days)

print("Twoja fala fizyczna:", fizyczna)
print("Twoja fala emocjonalna:", emocjonalna)
print("Twoja fala intelektualna:", intelektualna)

if fizyczna > 0.5:
    print("Dzisiaj jest dobry dzień na ćwiczenia, gratuluje!")
elif fizyczna < -0.5:
    if physical_wave(days + 1) > fizyczna:
        print("Jutro bedzie lepiej fizycznie")
    else:
        print("Dzisiaj jest dobry dzień na odpoczynek")


if emocjonalna > 0.5:
    print("Dzisiaj jest dobry dzień na spotkanie z przyjaciółmi, gratuluje!")
elif emocjonalna < -0.5:

    if emotional_wave(days + 1) > emocjonalna:
        print("Jutro bedzie lepiej emocjonalnie")
    else:
        print(
            "Dzisiaj jest dobry dzień na samotne spędzenie czasu, twoja fala na jutro to: "
        )

if intelektualna > 0.5:
    print("Dzisiaj jest dobry dzień na naukę, gratuluje!")
elif intelektualna < -0.5:
    if intellectual_wave(days + 1) > intelektualna:
        print("Jutro bedzie lepiej intelektualnie")
    else:
        print(
            "Dzisiaj jest dobry dzień na odpoczynek od nauki, twoja fala na jutro to: "
        )

# c) Mniej więcej 10min
