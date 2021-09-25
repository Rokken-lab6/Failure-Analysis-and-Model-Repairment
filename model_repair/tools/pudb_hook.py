try:
    import sys, pudb, traceback

    def info(type, value, tb):
        traceback.print_exception(type, value, tb)
        pudb.pm()

    sys.excepthook = info
except:
    pass