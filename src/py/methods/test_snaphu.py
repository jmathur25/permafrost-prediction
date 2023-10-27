from isce.components.contrib.Snaphu.Snaphu import Snaphu
from isce.components import isceobj

wrapName="/permafrost-prediction-shared-data/isce2_outputs/jatin_ALPSRP021272170/raw_slc_slice.flat"
outName="/permafrost-prediction-shared-data/isce2_outputs/jatin_ALPSRP021272170/raw_slc_slice.unw"

snp = Snaphu()
snp.setInitOnly(True)
snp.setInput(wrapName)
snp.setOutput(outName)
snp.setWidth(357)
snp.setCostMode('SMOOTH')
snp.setEarthRadius(6392620.374037754)
snp.setWavelength(0.2360571)
snp.setAltitude(705265.7225786864)
# snp.setCorrfile(corName)
snp.setInitMethod('MCF')
#snp.setCorrLooks(corrLooks)
snp.setMaxComponents(20)
snp.setDefoMaxCycles(2)
snp.setRangeLooks(1)
snp.setAzimuthLooks(1)
# snp.setCorFileFormat('FLOAT_DATA')
snp.prepare()
snp.unwrap()

outImage = isceobj.Image.createUnwImage()
outImage.setFilename(outName)
outImage.setWidth(357)
outImage.setAccessMode('read')
outImage.finalizeImage()
outImage.renderHdr()
