#!/bin/bash

ps=disps.ps
infile=geonet_static_displacements/events5551.csv
#bounds=-R163/179.5/-47.5/-34
bounds=-R176/178/-38.5/-37.5
proj=-JM10c

gmt makecpt -Cpolar -T-0.2/0.2/0.1 -Z > vert_cpt.cpt
#awk -F"," '{print $2,$3,$6}' $infile

gmt pscoast -Da $bounds $proj -W -BNeWS -Bx100 -By100 -K -Lx9/1+c38+w20k > $ps
awk -F"," '{print $2,$3,$5,$6}' $infile | gmt psvelo $bounds $proj -Sn5/-1/-10 -Wthick -L -K -O >> $ps
awk -F"," '{print $2,$3,$5,$6,0.1,0.1,0.1,$4}' $infile | gmt psvelo $bounds $proj -Se5/-1/-10 -Wthick -Gblack -L -K -O >> $ps
awk -F"," '{print $2,$3,$7}' $infile | gmt psxy $bounds $proj -Sc0.2 -W -Cvert_cpt.cpt -K -O >> $ps
cmbounds="-JX100c -R0/100/0/100"
echo "5 5.8 0.2 0" | gmt psvelo $cmbounds -Sn5/-1/-10 -Wthick -L -K -O >> $ps
echo "5 5.8 0.2 0 0.1 0.1 0.1" | gmt psvelo $cmbounds -Se5/-1/-10 -Wthick -Gblack -L -K -O >> $ps

gmt psscale -Dx10.2/3.3+w5/0.3+jML -Cvert_cpt.cpt -B+l"Uplift (m)" -O >> $ps
gmt psconvert $ps -Tg -A -P


