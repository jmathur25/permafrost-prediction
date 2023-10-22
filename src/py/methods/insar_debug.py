from isce.applications.insarApp import Insar

# stripmapApp_ALPSRP017332170_ALPSRP016671410.xml --steps --dostep=misregistration
insar = Insar(name="stripmapApp", cmdline='/permafrost-prediction-shared-data/isce2_outputs/ALPSRP017332170_ALPSRP016671410/stripmapApp_ALPSRP017332170_ALPSRP016671410.xml --steps --dostep=misregistration')
insar.configure()
insar.run()
