def reorder_documents(docs):
    """
    Reorder documents to reduce the 'lost in the middle'
    effect.
    Example:
    Original:
        [D1, D2, D3, D4, D5]
    Reordered:
        [D1, D3, D5, D4, D2]
    """
    if len(docs) <= 2:
        return docs
    reordered = []
    left = 0
    right = len(docs) - 1
    while left <= right:
        reordered.append(docs[left])
        left += 1
        if left <= right:
            reordered.append(docs[right])
            right -= 1
    return reordered
