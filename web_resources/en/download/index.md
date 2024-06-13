# Egaroucid Download

There are Install version and Portable version. With Install version, you should just download the installer and execute it to install. Installer automatically selects the optimal revision for your environment. With Portable version, you should download a zip file and unzip it.



## Install Version

Please download an installer from this button. Then run the installer and install it!

<div class="download_button_container"><a class="download_button_a" href="REPLACE_DOWNLOAD_BUTTON_URL" aria-label="Download Egaroucid">
<div class="download_button"><img height="22pt" src="img/download.png" alt="download icon">Download Egaroucid REPLACE_DOWNLOAD_BUTTON_VERSION</div>
</a></div>

Admin right is needed to install. If you see "Windows protected your PC" popup, then you can run it with clicking "More info", then "Run anyway". Egaroucid has no malicious codes, but please do this  operation at your own risk. The images are examples in Japanese.

<div class="centering_box">
    <img class="pic2" src="img/cant_run1.png" alt="running accross Windows protected your PC">
    <img class="pic2" src="img/cant_run2.png" alt="Press More Info and then you can run Egaroucid">
</div>



Egaroucid GUI version is for only Windows 64 bit on CPUs made by Intel or AMD.

Egaroucid has some revisions (SIMD / Generic / AVX512) to optimize the speed for several environments. Installer automatically select the best revision for you.

Please visit [GitHub Releases](https://github.com/Nyanyan/Egaroucid/releases) to see older versions and release notes.



## Portable Version

<b>Egaroucid GUI version is for only Windows 64 bit on CPUs made by Intel or AMD.</b> If you are using windows with ARM CPU, you cannot use Egaroucid.

Please download the one which is suitable to your environment, and unzip it. Egaroucid_[version].exe is the executable of Egaroucid.



GUI_DOWNLOAD_TABLE_HERE



Egaroucid is optimized to SIMD version, which requires AVX2 instructions, but old CPUs (created in 2013 or older) might not be able to run it. If so, please install Generic version. If you have a CPU that have AVX-512 extensions, AVX512 edition may be faster.


Please visit [GitHub Releases](https://github.com/Nyanyan/Egaroucid/releases) to see older versions and release notes.



## About Changing Book Format

### to egbk3 Format

Book format is changed in Egaroucid 6.5.1 to ```.egbk3``` extension. If you used older version and install 6.5.1, Egaroucid automatically convert book format in first execution. Also, you can use old ```.egbk2``` and ```.egbk``` book in "Book Import" function.

### to egbk2 Format

Book format is changed in Egaroucid 6.3.0. The new book's filename extension is ```.egbk2```. If you used version 6.2.0 or older and install 6.3.0, Egaroucid automatically convert book format in first execution. Also, you can use old ```.egbk``` book in "Book Reference" and "Book merge", but the save format is only new ```.egbk2```.

