sudo apt install gparted
sudo gparted
sudo apt install mount
sudo mkdir /ssdsata
sudo vim /etc/fstab
#/dev/root		/	ext4	defaults		0	1
#/dev/sda1	/ssddata	ext4	defaults
sudo mount /ssdsata