import sys
try:
    import chefshatgym
    print('OK')
except Exception as e:
    import traceback
    traceback.print_exc()