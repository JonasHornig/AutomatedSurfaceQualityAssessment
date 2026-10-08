def AppendDictionary(Dictionary, Key, DataType):
    if not Key in Dictionary:
        if DataType == "dict":
            Dictionary[Key] = {}
        if DataType == "list":
            Dictionary[Key] = []
    else:
        print("WARNING! - Trying to create already existing key")
        print(f"Dictionary: {Dictionary} - Key: {Key}")

def ConfirmEqualLength(List1, List2):
    if len(List1) != len(List2):
        raise ValueError("SANITY CHECK ERROR! - Length of two lists should match, but doesn't.")
    if len(List1) == 0:
        raise ValueError("SANITY CHECK ERROR! - Got empty list")