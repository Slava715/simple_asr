from yargy import rule, or_
from yargy.pipelines import morph_pipeline, caseless_pipeline
from yargy.interpretation import fact, const
from yargy.predicates import eq, caseless, normalized, type

Number = fact('Number', ['int', 'multiplier'])
NUMS_RAW = {
    'ноль': 0,
    'нуль': 0,
    'нулевой': '0_',
    
    'один': 1,
    'первый': '1_',
    
    'два': 2,
    'второй': '2_',
    
    'три': 3,
    'третий': '3_',
    
    'четыре': 4,
    'четвертый': '4_',
    
    'пять': 5,
    'пятый': '5_',
    
    'шесть': 6,
    'шестой': '6_',
    
    'семь': 7,
    'седьмой': '7_',
    
    'восемь': 8,
    'восьмой': '8_',
    
    'девять': 9,
    'девятый': '9_',
    
    'десять': 10,
    'десятый': '10_',
    
    'одиннадцать': 11,
    'одиннадцатый': '11_',
    
    'двенадцать': 12,
    'двенадцатый': '12_',
    
    'тринадцать': 13,
    'тринадцатый': '13_',
    
    'четырнадцать': 14,
    'четырнадцатый': '14_',
    
    'пятнадцать': 15,
    'пятнадцатый': '15_',
    
    'шестнадцать': 16,
    'шестнадцатый': '16_',
    
    'семнадцать': 17,
    'семнадцатый': '17_',
    
    'восемнадцать': 18,
    'восемнадцатый': '18_',
    
    'девятнадцать': 19,
    'девятнадцатый': '19_',
    
    'двадцать': 20,
    'двадцатый': '20_',
    
    'тридцать': 30,
    'тридцатый': '30_',
    
    'сорок': 40,
    'сороковой': '40_',
    
    'пятьдесят': 50,
    'пятидесятый': '50_',
    
    'шестьдесят': 60,
    'шестидесятый': '60_',
    
    'семьдесят': 70,
    'семидесятый': '70_',
    
    'восемьдесят': 80,
    'восьмидесятый': '80_',
    
    'девяносто': 90,
    'девяностый': '90_',
    
    'сто': 100,
    'сотый': '100_',
    
    'двести': 200,
    'двухсотый': '200_',
    
    'триста': 300,
    'трехсотый': '300_',
    
    'четыреста': 400,
    'четырехсотый': '400_',
    
    'пятьсот': 500,
    'пятисотый': '500_',
    
    'шестьсот': 600,
    'шестисотый': '600_',
    
    'семьсот': 700,
    'семисотый': '700_',
    
    'восемьсот': 800,
    'восьмисотый': '800_',
    
    'девятьсот': 900,
    'девятисотый': '900_',
    
    'тысяча': 10**3,
    
    'миллион': 10**6,
    
    'миллиард': 10**9,
    
    'триллион': 10**12,
}
DOT = eq('.')
INT = type('INT')
THOUSANDTH = rule(caseless_pipeline(['тысячных', 'тысячная'])).interpretation(const(10**-3))
HUNDREDTH = rule(caseless_pipeline(['сотых', 'сотая'])).interpretation(const(10**-2))
TENTH = rule(caseless_pipeline(['десятых', 'десятая'])).interpretation(const(10**-1))
THOUSAND = or_(
    rule(caseless('т'), DOT),
    rule(caseless('тыс'), DOT.optional()),
    rule(normalized('тысяча')),
    rule(normalized('тыща'))
).interpretation(const(10**3))
MILLION = or_(
    rule(caseless('млн'), DOT.optional()),
    rule(normalized('миллион'))
).interpretation(const(10**6))
MILLIARD = or_(
    rule(caseless('млрд'), DOT.optional()),
    rule(normalized('миллиард'))
).interpretation(const(10**9))
TRILLION = or_(
    rule(caseless('трлн'), DOT.optional()),
    rule(normalized('триллион'))
).interpretation(const(10**12))
MULTIPLIER = or_(
    THOUSANDTH,
    HUNDREDTH,
    TENTH,
    THOUSAND,
    MILLION,
    MILLIARD,
    TRILLION
).interpretation(Number.multiplier)
NUM_RAW = rule(morph_pipeline(NUMS_RAW).interpretation(Number.int.normalized().custom(NUMS_RAW.get)))
NUM_INT = rule(INT).interpretation(Number.int.custom(int))
NUM = or_(
    NUM_RAW,
    NUM_INT
).interpretation(Number.int)
NUMBER = or_(
    rule(NUM, MULTIPLIER.optional())
).interpretation(Number)
