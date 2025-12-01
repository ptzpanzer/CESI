from osgeo import gdal, osr
from pyproj import Proj, transform


def cut_map(in_path, files, out_path):
    driver = gdal.GetDriverByName('GTiff')
    data = []
    for file in files:
        filename = in_path + file
        dataset = gdal.Open(filename)
        band = dataset.GetRasterBand(1)
        gtf = dataset.GetGeoTransform()

        in_proj = Proj(init='epsg:4326')
        out_proj = Proj(init='epsg:32632')
        p1_in = (10.7992, 48.421)
        p1 = transform(in_proj, out_proj, p1_in[0], p1_in[1])

        xOrigin = gtf[0]
        yOrigin = gtf[3]
        pixelWidth = gtf[1]
        pixelHeight = -gtf[5]

        i1 = int((p1[0] - xOrigin) / pixelWidth)
        j1 = int((yOrigin - p1[1]) / pixelHeight)

        new_cols = 1400
        new_rows = 1400

        data.append(band.ReadAsArray(i1, j1, new_cols, new_rows))

        new_x = xOrigin + i1 * pixelWidth
        new_y = yOrigin - j1 * pixelHeight
        new_transform = (new_x, gtf[1], gtf[2], new_y, gtf[4], gtf[5])

    output_file = out_path
    dst_ds = driver.Create(output_file,
                           new_cols,
                           new_rows,
                           len(files),
                           gdal.GDT_Float32)
    # writing output raster
    for i in range(len(files)):
        dst_ds.GetRasterBand(i+1).WriteArray(data[i])

    # setting extension of output raster
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(new_transform)
    wkt = dataset.GetProjection()
    # setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())

    # Close output raster dataset
    dataset = None
    dst_ds = None
