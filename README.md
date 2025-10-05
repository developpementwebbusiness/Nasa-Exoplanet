# 🚀 NASA Exoplanet App — Installation Guide

Follow these steps to install and run the NASA Exoplanet application using **Docker**.

---

## 🧩 Prerequisites

Before you begin, ensure you have **Docker Desktop** installed on your machine.

### 🐳 Install Docker Desktop

1. Go to the official Docker Desktop page:
   👉 [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

2. Click on **“Download Docker Desktop”** and select your operating system.

   <img width="2467" height="1085" alt="Docker Desktop Download" src="https://github.com/user-attachments/assets/617210e2-ae7b-4319-8a23-b080aa27cdf3" />

---

## 📥 Clone the Repository

You can either **clone** or **download** the project.

### Option 1 — Download ZIP

1. Go to the top of the github repository and click on the code icon.
2. Click on **Code → Download ZIP**.
3. Unzip the folder in your desired location, for example:

   ```
   C:\Nasa-Exoplanet
   ```

### Option 2 — Clone via Git
Alternatively, you can clone the repository manually:
Open your terminal or command prompt and run:

```bash
git clone https://github.com/developpementwebbusiness/Nasa-Exoplanet.git
```
⚠️ Note: You must have Git installed on your system to use this command.
If you don’t have it yet, download and install it from the official website:

## ▶️ Launching the App

You can launch the app directly by navigating into the Nasa-Exoplanet folder and double-clicking the file:

app_launcher.bat

⚠️ Windows only: The APP_launcher.bat file is a Windows batch script and will not work on Linux or macOS systems.

---

If it fails, try the following manual steps below 👇

## 💻 Open the Project Directory

Open your **Command Prompt** (Windows) or **Terminal** (Linux/Mac).

Example on Windows:

<img width="987" height="720" alt="Command Prompt Example" src="https://github.com/user-attachments/assets/bed34495-3864-405e-a640-0d325936fb51" />

Navigate to the directory where you cloned or extracted the project for exemple :

```bash
cd C:\Nasa-Exoplanet
```

---

## ⚙️ Build and Launch the Application

Once inside the project directory, run the following command to build and start the app using Docker Compose.

### 🪟 On Windows

```bash
docker compose up --build
```
With everything done it should look like this in the **Command Prompt** (Windows) :

<img width="2326" height="542" alt="Docker Compose Windows" src="https://github.com/user-attachments/assets/8e596af6-17f1-4d9d-8eed-f54de7004c69" />

### 🐧 On Linux

```bash
sudo docker compose up --build
```

---

## 🌐 Access the Application

After the build and launch process completes successfully, open your browser and go to:

🔗 **[http://localhost:3000](http://localhost:3000)**

If your network allows it, you can also access it via by searching in the cmd prompt for a second different address usually located near the previous address.

---

## ✅ You’re All Set!

Your NASA Exoplanet app should now be up and running!
Enjoy exploring the universe 🌌
