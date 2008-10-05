#!/bin/bash

while [ 1 ]; do
    # check for new messages
    # note that we set the email account password in ~/.fetchmailrc
    new_messages=$(fetchmail --check --protocol IMAP --ssl --username spt175 \
	hecky.it.northwestern.edu | sed 's/(//' | awk '{print $1-$3}')
    
    # now, blink many times before rechecking email messages
    for j in $(seq 1 100); do
	# pause between each blink pattern
	sleep 1
 	
        # blink LEDS once for each new message
	for i in $(seq 1 $new_messages); do
	    xset led 3
	    sleep 0.1
	    xset -led 3
	    sleep 0.1
	done

    done
done
