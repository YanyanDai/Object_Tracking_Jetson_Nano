import serial
import time

def initconnection(portNo, baudRate):
    try:
        ser = serial.Serial(portNo, baudRate)
        print("Device Connected")
        return ser
    except:
        print("Not connected")

def sendData(se,data,digits):
    myString= "$"
    for d in data:
        myString += str(d).zfill(digits)
    try:
        se.write(myString.encode())
        print(myString)
    except:
        print("Data transmission failed")
    

if __name__ == "__main__":
    ser = initconnection("/dev/ttyACM0", 115200)
    while True:
        sendData(ser,[0,255],4)
        time.sleep(1)
        sendData(ser,[0,0],4)
        time.sleep(1)