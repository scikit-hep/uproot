class First_Key(object):
    
    def __init__(self, pointer, fKeylen):
        self.fVersion = 4
        self.fObjlen = 370
        self.fDatime = 1573188772
        self.fKeylen = fKeylen
        self.fNbytes = self.fObjlen + self.fKeylen
        self.fCycle = 1
        self.fSeekKey = pointer.index
        self.fSeekPdir = 100
        self.packer = ">ihiIhhii"
        self.fClassName = b'TList'
        self.fName = b'StreamerInfo'
        self.fTitle = b'Doubly linked list'
        
    def values(self):
        return self.packer, self.fNbytes, self.fVersion, self.fObjlen, self.fDatime, self.fKeylen, self.fCycle, self.fSeekKey, self.fSeekPdir, self.fClassName, self.fName, self.fTitle
    