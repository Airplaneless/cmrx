import os
import io
import h5py
import numpy as np
import gzip
import torch
import nibabel as nib
from sqlitedict import SqliteDict
from torch.utils.data import Dataset, DataLoader


def get_file_name(dataset: str, subdataset: str):
    if dataset == 'Cine':
        prefix = 'cine_'
        if subdataset == 'lax':
            fname = prefix + 'lax.mat'
            fname_mask = prefix + 'lax_mask.mat'
            return (fname, fname_mask)
        elif subdataset == 'sax':
            fname = prefix + 'sax.mat'
            fname_mask = prefix + 'sax_mask.mat'
            return (fname, fname_mask)
        else:
            raise ValueError
    elif dataset == 'Mapping':
        prefix = 'T'
        if subdataset == 'T1':
            fname = prefix + '1map.mat'
            fname_mask = prefix + '1map_mask.mat'
            return (fname, fname_mask)
        elif subdataset == 'T2':
            fname = prefix + '2map.mat'
            fname_mask = prefix + '2map_mask.mat'
            return (fname, fname_mask)
        else:
            raise ValueError
    else:
        raise ValueError


def get_nii_file_name(dataset: str, subdataset: str):
    if dataset == 'Cine':
        prefix = 'cine_'
        if subdataset == 'lax':
            fname = prefix + 'lax_forlabel.nii.gz'
            fname_mask = prefix + 'lax_label.nii.gz'
            return (fname, fname_mask)
        elif subdataset == 'sax':
            fname = prefix + 'sax_forlabel.nii.gz'
            fname_mask = prefix + 'sax_label.nii.gz'
            return (fname, fname_mask)
        else:
            raise ValueError
    elif dataset == 'Mapping':
        prefix = 'T'
        if subdataset == 'T1':
            fname = prefix + '1map_forlabel.nii.gz'
            fname_mask = prefix + '1map_label.nii.gz'
            return (fname, fname_mask)
        elif subdataset == 'T2':
            fname = prefix + '2map_forlabel.nii.gz'
            fname_mask = prefix + '2map_label.nii.gz'
            return (fname, fname_mask)
        else:
            raise ValueError
    else:
        raise ValueError


class CMRSegDataset(Dataset):

    def __init__(self, dpath, subset) -> None:
        super().__init__()
        self.subset = subset
        self.dpath = dpath
        self.files = {}
        _, dataset, _, _ = self.dpath.split('/')[-4:]
        fname, fname_seg = get_nii_file_name(dataset, subset)
        for dp in sorted(os.listdir(self.dpath)):
            if os.path.isdir(os.path.join(dpath, dp)) and dp.startswith('P'):
                if os.path.isfile(os.path.join(dpath, dp, fname)) and os.path.isfile(os.path.join(dpath, dp, fname_seg)):
                    self.files[dp] = {
                        'img': os.path.join(dpath, dp, fname),
                        'seg': os.path.join(dpath, dp, fname_seg)
                    }
        self.keys = list(self.files)
    
    def __len__(self):
        return len(self.keys)

    def getbykey(self, key):
        fnames = self.files[key]
        # load img
        img = nib.load(fnames['img']).get_fdata()
        img = torch.from_numpy(img).permute(2,1,0)[None].float()
        seg = nib.load(fnames['seg']).get_fdata()
        seg = torch.from_numpy(seg).permute(2,1,0)[None].long()
        return img, seg

    def __getitem__(self, item):
        return self.getbykey(self.keys[item])


class CMRDataset(Dataset):

    def __init__(self, dpath, subset, load_all=False) -> None:
        super().__init__()
        self.load_all = load_all
        self.subset = subset
        self.dpath = dpath
        isCoil, dataset, _, acc = self.dpath.split('/')[-4:]
        isCoil = bool(isCoil == 'MultiCoil')
        self.isCoil = isCoil
        fname, fname_mask = get_file_name(dataset, self.subset)
        sname = fname.split('.')[0] + '_coilmap.pt'
        self.files = {}
        for dp in sorted(os.listdir(self.dpath)):
            if os.path.isdir(os.path.join(dpath, dp)) and dp.startswith('P'):
                if acc == 'FullSample':
                    if os.path.isfile(os.path.join(dpath, dp, fname)):
                        self.files[dp] = {
                            'ks': os.path.join(dpath, dp, fname),
                            'mask': None if acc == 'FullSample' else os.path.join(dpath, dp, fname_mask),
                            'coilmap': None,
                        }
                else:
                    if os.path.isfile(os.path.join(dpath, dp, fname_mask)):
                        self.files[dp] = {
                            'ks': os.path.join(dpath, dp, fname),
                            'mask': None if acc == 'FullSample' else os.path.join(dpath, dp, fname_mask),
                            'coilmap': None,
                        }
        self.keys = list(self.files)
    
    def __len__(self):
        return len(self.keys)

    def getbykey(self, key, skip_ks_load=False):
        fnames = self.files[key]
        # load ks
        if not skip_ks_load:
            with h5py.File(fnames['ks']) as f:
                hkey = list(f.keys())[0]
                # TODO: кек, костыль
                if self.load_all:
                    ks = torch.from_numpy(f[hkey][:].view(np.complex64))
                else:
                    # load only first time step for speedup
                    ks = torch.from_numpy(f[hkey][:1].view(np.complex64))
                ks = ks[:, :, :].cfloat() if self.isCoil else ks[:, :, None].cfloat()
        else:
            # TODO: кек, костыль
            # with h5py.File(fnames['ks']) as f:
                # hkey = list(f.keys())[0]
            ks = torch.zeros(1)
        # load mask
        if fnames['mask'] is not None:
            with h5py.File(fnames['mask']) as f:
                hkey = list(f.keys())[0]
                mask = torch.from_numpy(f[hkey][:]).float()
        else:
            mask = torch.ones(ks.shape[-2], ks.shape[-1])
        return ks, mask

    def __getitem__(self, item):
        return self.getbykey(self.keys[item])


def get_file_shape(fpath):
    with h5py.File(fpath) as f:
        hkey = list(f.keys())[0]
        shape = f[hkey].shape
    return shape


class CMRDataset2D(Dataset):

    def __init__(self, dpath, subset) -> None:
        super().__init__()
        self.subset = subset
        self.dpath = dpath
        isCoil, dataset, _, acc = self.dpath.split('/')[-4:]
        isCoil = bool(isCoil == 'MultiCoil')
        self.isCoil = isCoil
        fname, fname_mask = get_file_name(dataset, self.subset)
        self.files = {}
        for dp in sorted(os.listdir(self.dpath)):
            if os.path.isdir(os.path.join(dpath, dp)) and dp.startswith('P'):
                if os.path.isfile(os.path.join(dpath, dp, fname)):
                    fshape = get_file_shape(os.path.join(dpath, dp, fname))
                    for j in range(fshape[1]):
                        self.files[f'{dp}_{j}'] = {
                            'ks': os.path.join(dpath, dp, fname),
                            'mask': None if acc == 'FullSample' else os.path.join(dpath, dp, fname_mask),
                            'coilmap': None,
                            'nslice': j
                        }
        self.keys = list(self.files)
    
    def __len__(self):
        return len(self.keys)

    def getbykey(self, key, skip_ks_load=False):
        fnames = self.files[key]
        nslice = self.files[key]['nslice']
        # load ks
        if not skip_ks_load:
            with h5py.File(fnames['ks']) as f:
                hkey = list(f.keys())[0]
                ks = torch.from_numpy(f[hkey][:, nslice:nslice+1].view(np.complex64))
                ks = ks[:, :, :].cfloat() if self.isCoil else ks[:, :, None].cfloat()
        else:
            with h5py.File(fnames['ks']) as f:
                hkey = list(f.keys())[0]
                ks = torch.zeros(f[hkey].shape)[:, nslice:nslice+1]
        # load mask
        if fnames['mask'] is not None:
            with h5py.File(fnames['mask']) as f:
                hkey = list(f.keys())[0]
                mask = torch.from_numpy(f[hkey][:]).float()
        else:
            mask = torch.ones(ks.shape[-2], ks.shape[-1])
        return ks, mask

    def __getitem__(self, item):
        return self.getbykey(self.keys[item])


def t2b(x: torch.Tensor) -> bytes:
    xb = io.BytesIO()
    torch.save(x, xb)
    return gzip.compress(xb.getvalue(), compresslevel=0)

def b2t(xb: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(initial_bytes=gzip.decompress(xb)))


class PairDataset(Dataset):

    def __init__(self, dataset_gt: CMRDataset, dataset_hat: CMRDataset, isCached=False) -> None:
        super().__init__()
        self.dataset_gt = dataset_gt
        self.dataset_hat = dataset_hat
        assert self.dataset_gt.keys == self.dataset_hat.keys
        self.keys = self.dataset_gt.keys
        self.cache = dict()
        self.isCached = isCached
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]
        if key in self.cache:
            ks_gt, mask = self.cache[key]
        else:
            ks_gt, _ = self.dataset_gt.getbykey(key)
            _, mask = self.dataset_hat.getbykey(key, skip_ks_load=True)
            if self.isCached:
                self.cache[key] = [ks_gt, mask]
        return ks_gt * mask, ks_gt, mask


class PairSegDataset(Dataset):

    def __init__(self, dataset_gt: CMRDataset, dataset_seg: CMRSegDataset, isCached=False) -> None:
        super().__init__()
        self.dataset_gt = dataset_gt
        self.dataset_seg = dataset_seg
        assert self.dataset_gt.keys == self.dataset_seg.keys
        self.keys = self.dataset_gt.keys
        self.cache = dict()
        self.isCached = isCached
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]
        if key in self.cache:
            ks_gt, seg = self.cache[key]
        else:
            ks_gt, _ = self.dataset_gt.getbykey(key)
            _, seg = self.dataset_seg.getbykey(key)
            if self.isCached:
                self.cache[key] = [ks_gt, seg]
        return ks_gt, seg


class TripleSegDataset(Dataset):

    def __init__(self, dataset_gt: CMRDataset, dataset_hat: CMRDataset, dataset_seg: CMRSegDataset, isCached=False) -> None:
        super().__init__()
        self.dataset_gt = dataset_gt
        self.dataset_hat = dataset_hat
        self.dataset_seg = dataset_seg
        assert self.dataset_gt.keys == self.dataset_hat.keys == self.dataset_seg.keys
        self.keys = self.dataset_gt.keys
        self.cache = dict()
        self.isCached = isCached
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]
        if key in self.cache:
            ks_gt, mask, seg = self.cache[key]
        else:
            ks_gt, _ = self.dataset_gt.getbykey(key)
            _, mask = self.dataset_hat.getbykey(key, skip_ks_load=True)
            _, seg = self.dataset_seg.getbykey(key)
            if self.isCached:
                self.cache[key] = [ks_gt, mask, seg]
        return ks_gt * mask, ks_gt, mask, seg


class DatasetDynamicStorage(torch.utils.data.Dataset):

    def __init__(self, path_db) -> None:
        super().__init__()
        self.db_path = path_db
        db = SqliteDict(self.db_path)
        while db['state'] != 0:
            pass
        db.close()
        print('[DatasetDynamicStorage ready]')

    def __len__(self):
        db = SqliteDict(self.db_path)
        res = len(db['index'])
        db.close()
        return res

    def __getitem__(self, item: int):
        db = SqliteDict(self.db_path, autocommit=True)
        while db[item]['state'] != 0:
            pass
        rec = db[item]
        path = rec['path']
        res = torch.load(path, map_location='cpu')
        rec['used'] = 1
        db[item] = rec
        db.close()
        return res


