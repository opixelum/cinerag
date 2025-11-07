def small2big(chunks, idx, window=1):
    """
    Retourne une string contenant le chunk sélectionné (idx) et ses voisins dans une fenêtre de taille 'window'.
    Exemple : small2big(chunks, idx=5, window=1) retourne 'chunk4\n\nchunk5\n\nchunk6'
    """
    start = max(0, idx - window)
    end = min(len(chunks), idx + window + 1)
    return "\n\n".join(chunks[start:end])