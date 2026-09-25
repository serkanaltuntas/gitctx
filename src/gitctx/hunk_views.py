"""Clean old/new hunk text and separately auditable original row correspondence."""
from gitctx.student_sequences import diff_units, physical_lines


def prepare(diff):
    lines=physical_lines(diff);views=[];maps=[]
    for unit_id,u in enumerate(diff_units(diff)):
        if u['kind']!='hunk':
            views.append({'id':unit_id,'metadata':''.join(lines[u['start_line']:u['end_line']])})
            maps.extend((i,lines[i]) for i in range(u['start_line'],u['end_line']))
            continue
        before=[];after=[];metadata=[]
        for i in range(u['start_line'],u['end_line']):
            line=lines[i];maps.append((i,line))
            if line.startswith((' ','-')):before.append(line[1:])
            if line.startswith((' ','+')):after.append(line[1:])
            if not line.startswith((' ','-','+')):metadata.append(line)
        views.append({'id':unit_id,'file':u['file'],'hunk_header':''.join(metadata),
                      'before':''.join(before),'after':''.join(after)})
    if [i for i,_ in maps]!=list(range(len(lines))) or ''.join(s for _,s in maps)!=diff:
        raise ValueError('source correspondence lost')
    return views
