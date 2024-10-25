import ms5803py
import time

# Constants
SEA_LEVEL_PRESSURE = 1013.25  # in mBar


def calculate_altitude(pressure, sea_level_pressure=SEA_LEVEL_PRESSURE):
    return 44330 * (1 - (pressure / sea_level_pressure) ** 0.1903)


s = ms5803py.MS5803()

while True:
    # Quick and easy reading
    # press, temp = s.read(pressure_osr=512)
    # altitude = calculate_altitude(press)
    # print(f"Quick'n'easy pressure={press} mBar, temperature={temp} C, altitude={altitude} meters")

    # Advanced reading
    raw_temperature = s.read_raw_temperature(osr=4096)
    for i in range(5):
        raw_pressure = s.read_raw_pressure(osr=256)
        press, temp = s.convert_raw_readings(raw_pressure, raw_temperature)
        altitude = calculate_altitude(press)

    print(f"Advanced pressure={press} mBar, temperature={temp} C, altitude={altitude} meters")

    time.sleep(0.1)