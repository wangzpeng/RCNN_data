import subprocess

p1=subprocess.call('../tools/demo.py')

p2=subprocess.call('python /home/wang/tf-faster-rcnn/tools/deal_roi.py',shell=True)

p3=subprocess.call('python /home/wang/CNN/test.py',shell=True)


