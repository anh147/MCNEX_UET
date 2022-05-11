from ctypes import CFUNCTYPE
from PyQt5.QtCore import QFileSelector
import snap7
import numpy as np

class PLC(object):

    def __init__(self):
        
        # Init Variables
        self.IP = '192.168.0.3'       # IP của PLC
        self.slot = 1                   # Lấy trong TIA Portal
        self.rack = 0                   # Lấy trong TIA Portal
        self.DBNumber = 39              # Data Block cần nhận dữ liệu (DB1, DB2,...)
        self.dataStart = 1              # Vị trí bit con trỏ nhận dữ liệu
        self.dataSize = 254             # Độ dài của data (1 byte, 4 bytes, 8 bytes,...)
        self.data = np.zeros(96)        # Biến truyền data cho PLC
    
    # Test Connection with PLC
    def testConnection(self):
        plc = snap7.client.Client()
        try:
            plc.connect(self.IP, self.rack, self.slot)
        except Exception as e:
            print("Connection Error!")
        finally:
            if plc.get_connected():
                plc.disconnect()
                print("Connection Success!")

    # Read Data Function
    def queryCommand(self):
        plc = snap7.client.Client()
        again = True
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                data = plc.db_read(self.DBNumber, 13, self.dataSize)
                again = False
                return snap7.util.get_string(data, -1, 254)
            except Exception as e:
                print("Cannot Get Command! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()

    def jig_Signal(self):
        plc = snap7.client.Client()
        again = True
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                data = plc.db_read(self.DBNumber, 269, 1)
                again = False
                return snap7.util.get_bool(data, 0, 1)
            except Exception as e:
                print("Cannot Get Signal! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()
    def status_cam_checked(self):
        plc = snap7.client.Client()
        again = True
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                data = plc.db_read(self.DBNumber, 271, self.dataSize)
                again = False
                return snap7.util.get_string(data, -1, 254)
            except Exception as e:
                print("Cannot Get Status Cam Checked! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()
    def status_cam_in_jig(self):
        plc = snap7.client.Client()
        again = True
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                data = plc.db_read(self.DBNumber, 536, self.dataSize)
                again = False
                return snap7.util.get_string(data, -1, 254)
            except Exception as e:
                print("Cannot Get Status Cam In Jig! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()
    # Write Data Function
    def sendCommand(self, command):
        plc = snap7.client.Client()
        again = True
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                data = plc.db_read(self.DBNumber, 13, self.dataSize)
                snap7.util.set_string(data, -1, command, self.dataSize)
                if not data.strip():
                    print("Command Corrupted!")
                    return
                plc.db_write(self.DBNumber, 13, data)
                print("Command Write Successfully!")
                again = False
            except Exception as e:
                print("Cannot Send Command! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()
    
    def sendData(self):
        plc = snap7.client.Client()
        again = True

        plc.connect(self.IP, self.rack, self.slot)
        
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                for i in range(96):
                    data = plc.db_read(self.DBNumber, 1+int(i/8), 1)
                    snap7.util.set_bool(data, 0,i%8, self.data[i])
                    plc.db_write(self.DBNumber, 1+int(i/8), data)
                    # print("147")
                print("Data Write Successfully!")
                again = False
            except Exception as e:
                print("Cannot Send Data! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()

    def sendTotal(self, total):
        plc = snap7.client.Client()
        again = True
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                data = plc.db_read(self.DBNumber, 806, 1)
                snap7.util.set_int(data, 1, total)
                plc.db_write(self.DBNumber, 806, data)
                print("Count Write Successfully!")
                again = False
            except Exception as e:
                print("Cannot Send Total! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()
    def send_status_cam_check(self, command):
        plc = snap7.client.Client()
        again = True
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                data = plc.db_read(self.DBNumber, 527, self.dataSize)
                snap7.util.set_string(data, -1, command, self.dataSize)
                if not data.strip():
                    print("Command Corrupted!")
                    return
                plc.db_write(self.DBNumber, 527, data)
                print("Status Cam Check Command Write Successfully!")
                again = False
            except Exception as e:
                print("Cannot Send Status Cam Check Command! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()
    def send_status_cam_inJig(self, command):
        plc = snap7.client.Client()
        again = True
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                data = plc.db_read(self.DBNumber, 527, self.dataSize)
                snap7.util.set_string(data, -1, command, self.dataSize)
                if not data.strip():
                    print("Command Corrupted!")
                    return
                plc.db_write(self.DBNumber, 527, data)
                print("Command Status Cam In Jig Write Successfully!")
                again = False
            except Exception as e:
                print("Cannot Send Status Cam In Jig Command! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()
    def sendSignal(self, coord, signal):
        plc = snap7.client.Client()
        again = True
        while again:
            try:
                plc.connect(self.IP, self.rack, self.slot)
                data = plc.db_read(self.DBNumber, 804, 1)
                snap7.util.set_bool(data, 0, coord, signal)
                plc.db_write(self.DBNumber, 804, data)
                # print("Count Write Successfully!")
                again = False
            except Exception as e:
                print("Cannot Send Signal! Error!")
            finally:
                if plc.get_connected():
                    plc.disconnect()

if __name__ == "__main__":
    Controller = PLC()
    Controller.testConnection()
    # Controller.sendData()
    
    # print(Controller.queryCommand())

    #Controller.Jig_Signal()
    # print(Controller.jig_Signal())
    
    command = Controller.queryCommand()
    print(Controller.queryCommand())
    # Controller.sendCommand('2')
    # print(Controller.queryCommand())

    # result = np.ones(96, dtype=int)
    # Controller.data = result
    # Controller.sendData()

    # Controller.send_status_cam_check('Ok')
    # print("Command = ", command)
    if command == 'Detect':

        result = np.zeros(96, dtype=int)
        check_yes = np.array([0, 1, 2, 3, 4])
                    
        for i in range(96):
            for j in range(check_yes.size):
                if i == check_yes[j]: 
                    result[i] = 1
        print(result)
        Controller.data = result

        print("Data: ", Controller.data.size)
        Controller.sendData()

    Controller.sendCommand('Done_detect')
    print(Controller.queryCommand())

    #print(Controller.CamCheckCommand())
    command = Controller.queryCommand()
    print("Command2 = ", command)
    
    if command == 'Check':
        check_align = 1
        if check_align:

            Controller.send_status_cam_inJig('Ok_for_jig')
            print(Controller.status_cam_in_jig())

            if Controller.jig_Signal:
                error_check = 'OK'
                Controller.send_status_cam_inJig(error_check)


        else:
            Controller.send_status_cam_inJig('Skeff')
            print(Controller.status_cam_in_jig())




        
    # print(Controller.status_cam_in_jig())
    # Controller.send_status_cam_inJig('3')
    # print(Controller.status_cam_in_jig())



