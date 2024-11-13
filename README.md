# fall-detection-phd

sudo apt install -y i2c-tools python3-pip

pip install pandas smbus numpy scikit-learn board Adafruit-Blinka adafruit-circuitpython-bmp3xx board --break-system-packages


To run the script as root, you can modify the service file to ensure it runs with root privileges. Here’s how:
1. Create the Service File (if not already done):
```commandline
sudo nano /etc/systemd/system/script.service
```
2. Edit the Service File: In the [Service] section, do not specify a User= line. This omission will make the script run as root by default. Your service file should look like this:
```commandline
[Unit]
Description=Python Script Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/ivanursul/script.py
Restart=always
WorkingDirectory=/home/ivanursul
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```
3. Reload systemd to recognize the updated service:
```commandline
sudo systemctl daemon-reload
```
4. Start the service
```commandline
sudo systemctl start script.service
```
5. Enable the service at boot: 
```commandline
sudo systemctl enable script.service

```
6. Verify the status:
```commandLine
sudo systemctl status script.service
```
7. 