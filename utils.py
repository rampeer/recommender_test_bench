def avg_update(old, count, next):
    return old * (float(count) - 1) / float(count) + next / float(count)
