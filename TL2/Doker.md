# 1ο Βήμα Εγκατάσταση του Docker 

![Screenshot 2024-05-26 154217](https://github.com/Metsk-Inc/TL2024/assets/100226126/931b2664-a9e1-4de9-a4c3-c4e0fc9815be)

# 2ο Βήμα Κατασκευή του Docker Image 
docker build -t streamlit-app .

![image](https://github.com/Metsk-Inc/TL2024/assets/100226126/b09e97b6-0002-4fee-a5b4-3f82f83a1313)



# 3ο Βήμα Είσοδος στις Ρυθμίσεις του Δρομολογητή
Χρησιμοποιήσαμε την διευθύνση 192.168.1.1
![gh](https://github.com/Metsk-Inc/TL2024/assets/100226126/46bc625a-d73d-47cd-b11b-067bfcfba85d)

# 4ο Βήμα Port Forwarding
Δημιουργήσαμε έναν κανόνα για τη θύρα 8501 που να προωθεί την κίνηση στην τοπική IP του υπολογιστή .
![24-05-26 165021](https://github.com/Metsk-Inc/TL2024/assets/100226126/f552e38e-7ce3-49dd-bf83-4773b8eb0b72)
# 5ο Βήμα Προσθήκη Νέου Κανόνα(Firewall)
Προσθέσαμε έναν κανόνα για την θύρα 8501
![ξκη6 165218](https://github.com/Metsk-Inc/TL2024/assets/100226126/b03729ae-192a-48eb-b2ce-78e14a8117cb)
![239](https://github.com/Metsk-Inc/TL2024/assets/100226126/4714acf8-a561-4bc4-bbc2-a5c6d7891306)
# 6ο Βήμα Εκτέλεση του Docker Container
docker run -p 8501:8501 streamlit-app
![Screenshot 2024-05-26 165405](https://github.com/Metsk-Inc/TL2024/assets/100226126/e8eae85a-e4f0-4739-aebd-06f50beb7f68)
