import smbus
import time

# MPU6050 and BMP180 addresses
MPU6050_ADDR = 0x68
BMP180_ADDR = 0x77

# MPU6050 Registers
MPU6050_PWR_MGMT_1 = 0x6B
MPU6050_ACCEL_XOUT_H = 0x3B
MPU6050_ACCEL_YOUT_H = 0x3D
MPU6050_ACCEL_ZOUT_H = 0x3F

# BMP180 Registers
BMP180_CONTROL = 0xF4
BMP180_TEMPDATA = 0xF6
BMP180_PRESSUREDATA = 0xF6
BMP180_READTEMPCMD = 0x2E
BMP180_READPRESSURECMD = 0x34

# I2C bus
bus = smbus.SMBus(1)


# Function to read raw data from MPU6050
def read_mpu6050_raw_data(register):
    high = bus.read_byte_data(MPU6050_ADDR, register)
    low = bus.read_byte_data(MPU6050_ADDR, register + 1)
    value = (high << 8) | low
    if value > 32767:
        value = value - 65536
    return value


# Initialize MPU6050
def initialize_mpu6050():
    bus.write_byte_data(MPU6050_ADDR, MPU6050_PWR_MGMT_1, 0)


# Read accelerometer data
def read_accelerometer_data():
    acc_x = read_mpu6050_raw_data(MPU6050_ACCEL_XOUT_H)
    acc_y = read_mpu6050_raw_data(MPU6050_ACCEL_YOUT_H)
    acc_z = read_mpu6050_raw_data(MPU6050_ACCEL_ZOUT_H)

    # Convert to g values (assuming range is +/- 2g, scale factor is 16384)
    acc_x = acc_x / 16384.0
    acc_y = acc_y / 16384.0
    acc_z = acc_z / 16384.0

    return acc_x, acc_y, acc_z


# Read calibration data from BMP180
def read_bmp180_calibration_data():
    cal = {}
    cal['AC1'] = read_signed_16_bit(BMP180_ADDR, 0xAA)
    cal['AC2'] = read_signed_16_bit(BMP180_ADDR, 0xAC)
    cal['AC3'] = read_signed_16_bit(BMP180_ADDR, 0xAE)
    cal['AC4'] = read_unsigned_16_bit(BMP180_ADDR, 0xB0)
    cal['AC5'] = read_unsigned_16_bit(BMP180_ADDR, 0xB2)
    cal['AC6'] = read_unsigned_16_bit(BMP180_ADDR, 0xB4)
    cal['B1'] = read_signed_16_bit(BMP180_ADDR, 0xB6)
    cal['B2'] = read_signed_16_bit(BMP180_ADDR, 0xB8)
    cal['MB'] = read_signed_16_bit(BMP180_ADDR, 0xBA)
    cal['MC'] = read_signed_16_bit(BMP180_ADDR, 0xBC)
    cal['MD'] = read_signed_16_bit(BMP180_ADDR, 0xBE)
    return cal


def read_signed_16_bit(address, register):
    high = bus.read_byte_data(address, register)
    low = bus.read_byte_data(address, register + 1)
    value = (high << 8) + low
    if value > 32767:
        value = value - 65536
    return value


def read_unsigned_16_bit(address, register):
    high = bus.read_byte_data(address, register)
    low = bus.read_byte_data(address, register + 1)
    value = (high << 8) + low
    return value


# Read temperature and calculate B5
def read_bmp180_temperature(cal):
    bus.write_byte_data(BMP180_ADDR, BMP180_CONTROL, BMP180_READTEMPCMD)
    time.sleep(0.005)  # Wait for converter
    UT = read_unsigned_16_bit(BMP180_ADDR, BMP180_TEMPDATA)

    # Calculate true temperature using calibration data
    X1 = ((UT - cal['AC6']) * cal['AC5']) // (1 << 15)
    X2 = (cal['MC'] * (1 << 11)) // (X1 + cal['MD'])
    B5 = X1 + X2
    temperature = (B5 + 8) // (1 << 4)

    return temperature / 10.0, B5  # In degrees Celsius


# Calculate true pressure
def read_bmp180_pressure(cal, B5):
    OSS = 3  # Oversampling setting
    bus.write_byte_data(BMP180_ADDR, BMP180_CONTROL, BMP180_READPRESSURECMD + (OSS << 6))
    time.sleep(0.026)  # Wait for converter
    MSB = bus.read_byte_data(BMP180_ADDR, BMP180_PRESSUREDATA)
    LSB = bus.read_byte_data(BMP180_ADDR, BMP180_PRESSUREDATA + 1)
    XLSB = bus.read_byte_data(BMP180_ADDR, BMP180_PRESSUREDATA + 2)
    UP = ((MSB << 16) + (LSB << 8) + XLSB) >> (8 - OSS)

    # Calculate true pressure using calibration data
    B6 = B5 - 4000
    X1 = (cal['B2'] * ((B6 * B6) // (1 << 12))) // (1 << 11)
    X2 = (cal['AC2'] * B6) // (1 << 11)
    X3 = X1 + X2
    B3 = (((cal['AC1'] * 4 + X3) << OSS) + 2) // 4  # Ensure integer division

    X1 = (cal['AC3'] * B6) // (1 << 13)
    X2 = (cal['B1'] * ((B6 * B6) // (1 << 12))) // (1 << 16)
    X3 = ((X1 + X2) + 2) // (1 << 2)
    B4 = (cal['AC4'] * (X3 + 32768)) // (1 << 15)
    B7 = (UP - B3) * (50000 >> OSS)

    if B7 < 0x80000000:
        pressure = (B7 * 2) // B4
    else:
        pressure = (B7 // B4) * 2

    X1 = (pressure // (1 << 8)) * (pressure // (1 << 8))
    X1 = (X1 * 3038) // (1 << 16)
    X2 = (-7357 * pressure) // (1 << 16)
    pressure = pressure + ((X1 + X2 + 3791) // (1 << 4))

    return pressure  # In Pa


# Calculate altitude from pressure
def calculate_altitude(pressure, sea_level_pressure=101325.0):
    # Altitude calculation based on the barometric formula
    altitude = 44330.0 * (1.0 - (pressure / sea_level_pressure) ** 0.1903)
    return altitude


# Main loop to read data
def main():
    initialize_mpu6050()
    print("Reading MPU6050 and BMP180 data...")

    # Read calibration data from BMP180
    cal = read_bmp180_calibration_data()

    # You can set the sea-level pressure for your location (in Pascals)
    local_sea_level_pressure = 101325.0  # Adjust this based on your local sea-level pressure

    while True:
        # MPU6050 data
        acc_x, acc_y, acc_z = read_accelerometer_data()
        print(f"Accelerometer (g): X={acc_x:.2f}, Y={acc_y:.2f}, Z={acc_z:.2f}")

        # BMP180 data
        temp, B5 = read_bmp180_temperature(cal)
        pressure = read_bmp180_pressure(cal, B5)
        altitude = calculate_altitude(pressure, local_sea_level_pressure)

        print(f"BMP180: Temp={temp:.2f}°C, Pressure={pressure:.2f} Pa, Altitude={altitude:.2f} meters")

        time.sleep(1)


if __name__ == "__main__":
    main()
