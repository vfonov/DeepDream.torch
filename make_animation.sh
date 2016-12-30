#! /bin/sh

convert -verbose -delay 50 -loop 0 moto_??.jpg -layers optimize  moto_anim_train.gif