import nice_model
import torch

def test_smart_spliter_odd_split():
    x = torch.tensor([[1,2,3,4,5,6]])
    spliter_odd = nice_model.BaseSpliter(False, 6)
    y, z = spliter_odd.split(x)
    assert torch.equal(y, torch.tensor([[2, 4, 6]])) and torch.equal(z, torch.tensor([[1, 3, 5]]))

def test_smart_spliter_merge():
    x = torch.tensor([[1,2,3,4,5,6]])
    spliter_odd = nice_model.BaseSpliter(False, 6)
    y, z = spliter_odd.split(x)
    res = spliter_odd.merge(y, z)
    print(res, type(res))
    assert torch.equal(x, res)

def test_complementary():
    x = torch.tensor([[1,2,3,4,5,6]])
    spliter_odd = nice_model.BaseSpliter(False, 6)
    y1, z1 = spliter_odd.split(x)
    spliter_even = nice_model.BaseSpliter(True, 6)
    y2, z2 = spliter_even.split(x)
    assert torch.equal(y1, z2) and torch.equal(z1, y2)
