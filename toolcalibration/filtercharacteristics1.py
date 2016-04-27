
class struct:
     def __init__(self, **kwds):
         self.__dict__.update(kwds)
 
filtercharact = []

fil_item = struct(designname = 'butter',
       Norder         = 2,
       Wlow_Hz        = 0.01,
       Whigh_Hz       = 0.06,
       SCPperiod_sec  = 1000,
       windowshape    = 'hann',
       overlapDFT     = 0.5,
       overlapSCP     = 0,
       ratioDFT2SCP   = 5)
filtercharact.append(fil_item)


fil_item = struct(designname = 'butter',
       Norder         = 2,
       Wlow_Hz        = 0.05,
       Whigh_Hz       = 0.15,
       SCPperiod_sec  = 700,
       windowshape    = 'hann',
       overlapDFT     = 0.5,
       overlapSCP     = 0,
       ratioDFT2SCP   = 5)
filtercharact.append(fil_item)

fil_item = struct(designname = 'butter',
       Norder         = 2,
       Wlow_Hz        = 0.14,
       Whigh_Hz       = 0.35,
       SCPperiod_sec  = 250,
       windowshape    = 'hann',
       overlapDFT     = 0.5,
       overlapSCP     = 0,
       ratioDFT2SCP   = 5)
filtercharact.append(fil_item)

fil_item = struct(designname = 'butter',
       Norder         = 2,
       Wlow_Hz        = 0.32,
       Whigh_Hz       = 0.7,
       SCPperiod_sec  = 150,
       windowshape    = 'hann',
       overlapDFT     = 0.5,
       overlapSCP     = 0,
       ratioDFT2SCP   = 5)
filtercharact.append(fil_item)

fil_item = struct(designname = 'butter',
       Norder         = 3,
       Wlow_Hz        = 0.6,
       Whigh_Hz       = 0.15,
       SCPperiod_sec  = 30,
       windowshape    = 'hann',
       overlapDFT     = 0.5,
       overlapSCP     = 0,
       ratioDFT2SCP   = 5)
filtercharact.append(fil_item)

fil_item = struct(designname = 'butter',
       Norder         = 5,
       Wlow_Hz        = 1.5,
       Whigh_Hz       = 6.0,
       SCPperiod_sec  = 20,
       windowshape    = 'hann',
       overlapDFT     = 0.5,
       overlapSCP     = 0,
       ratioDFT2SCP   = 5)
filtercharact.append(fil_item)

