from tipi_data.models.deputy import Deputy
from tipi_data.models.parliamentarygroup import ParliamentaryGroup


# Grouped deputies with name in correct order
class GroupedDeputies:
    def __init__(self):
        self.__deputies = Deputy.objects()
        self.__parliamentarygroups = ParliamentaryGroup.objects()
        self.grouped_deputies = self.__make_grouped_deputies()

    def get_deputies(self, groups=None):
        if groups is None or len(groups) == 0:
            return self.__get_all_deputies()
        filtered_deputies = list()
        for group in groups:
            filtered_deputies += self.__get_deputies_by_group(group)
        return filtered_deputies

    def __get_all_deputies(self):
        deputies = list()
        for key in self.grouped_deputies.keys():
            for deputy in self.grouped_deputies[key]:
                deputies.append(deputy)
        return deputies

    def __get_deputies_by_group(self, group):
        if group in self.grouped_deputies.keys():
            return self.grouped_deputies[group]

    def __make_grouped_deputies(self):
        grouped_deputies = dict()
        for group in self.__parliamentarygroups:
            grouped_deputies[group['name']] = list(filter(
                lambda d: d['parliamentarygroup'] == group['shortname'],
                self.__deputies))
        return grouped_deputies
