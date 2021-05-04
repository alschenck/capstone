import downloads.download_zips as down
import zips.extract_zips as zips
import csvParse.parseToCsv as parseCSV

if __name__ == '__main__':
    down.getAllData()
    zips.unzip()
    parseCSV.run()