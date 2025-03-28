from datetime import datetime
from datetime import timedelta

from tipi_data.repositories.initiatives import Initiatives


class FootprintQueryManager:

    def parse_query(self, types, topic, entity, typeof, status):
        query = {
                'initiative_type_alt': {'$in': types},
                'status': status,
                }
        if topic:
            query['tagged.topics'] = topic
        if typeof == 'deputy':
            query['author_deputies'] = entity
        if typeof == 'parliamentarygroup':
            query['author_parliamentarygroups'] = entity
        return query


class FootprintSumManager(FootprintQueryManager):

    def __init__(self, topic, entity, typeof):
        self.topic = topic
        self.entity = entity
        self.typeof = typeof

    def types(self):
        return list()

    def status(self):
        return {'$not': {'$in': ['No admitida a trámite', 'Retirada']}}

    def multiply(self):
        return 1

    def compute(self):
        query = self.parse_query(self.types(), self.topic, self.entity, self.typeof, self.status())

        if not self.topic:
            return Initiatives.by_query(query).count() * self.multiply()

        pipeline = []
        pipeline.append({ "$match": query })
        pipeline.append({ "$unwind": "$tagged" })
        pipeline.append({ "$unwind": "$tagged.topic_alignment" })
        pipeline.append({ "$match": {
                "tagged.topic_alignment.topic": self.topic }
            })
        pipeline.append({"$addFields": {
            "percentage_fraction": { "$divide": [ "$tagged.topic_alignment.percentage", 100 ] }
            }
        })

        def number_of_deputies():
            return { "$size": "$author_deputies" } if self.typeof == 'deputy' else 1

        pipeline.append({ "$addFields": {
            "count_deputies": number_of_deputies()
            }
        })

        pipeline.append({"$addFields": {
            "weighted_percentage": {
                "$cond": {
                    "if": { "$gt": [ "$count_deputies", 0 ] },
                    "then": { "$divide": [ "$percentage_fraction", "$count_deputies" ] },
                    "else": 0
                    }
                }
            }
        })

        pipeline.append({ "$group": {
                "_id": None,
                "output": { "$sum": "$weighted_percentage" }
                }
        })
        pipeline.append({ "$project": { "_id": 0, "output": 1} })

        result = list(Initiatives.get_all().aggregate(*pipeline))
        if not result:
            return 0
        return result[0]['output'] * self.multiply()



class FootprintSumPointOneManager(FootprintSumManager):
    def types(self):
        return [
                'Pregunta al Gobierno con respuesta escrita',
                'Pregunta a la Corporación RTVE con respuesta escrita',
                ]

    def multiply(self):
        return 0.1



class FootprintSumFourManager(FootprintSumManager):
    def types(self):
        return [
                'Comparecencia del Gobierno en Comisión (art. 44)',
                'Comparecencia del Gobierno en Comisión (arts. 202 y 203)',
                'Comparecencia de autoridades y funcionarios en Comisión',
                'Comparec. autoridades y funcionarios en Com. Mx. solicitada en Senado',
                'Otras comparecencias en Comisión',
                'Interpelación urgente',
                'Interpelación ordinaria',
                'Solicitud de informe a la Administración del Estado (art. 7)',
                'Solicitud de informe a otra Entidad Pública (art. 7)',
                'Solicitud de informe a la Administración del Estado (art. 44)',
                'Solicitud de informe a otra Entidad Pública (art. 44)',
                'Otras solicitudes de informe (art. 44)',

                ]
    def __multiply(self):
        return 4


class FootprintSumTenManager(FootprintSumManager):
    def types(self):
        return [
                'Pregunta oral en Pleno',
                'Pregunta oral al Gobierno en Comisión',
                'Moción de censura',
                'Pregunta oral a la Corporación RTVE',
                ]

    def multiply(self):
        return 10


class FootprintSumFourtyManager(FootprintSumManager):
    def types(self):
        return [
                'Proposición no de Ley ante el Pleno',
                'Proposición no de Ley en Comisión',
                'Moción consecuencia de interpelación ordinaria',
                'Moción consecuencia de interpelación urgente',
                ]

    def multiply(self):
        return 40


class FootprintSumEightyManager(FootprintSumManager):
    def types(self):

        return [
                'Proposición de ley de Grupos Parlamentarios del Congreso',
                'Proposición de ley de Diputados',
                'Proyecto de reforma Constitucional',
                'Proposición de reforma Constitucional de Grupos Parlamentarios',
                'Proposición de reforma constitucional de Comunidades Autónomas',
                ]

    def multiply(self):
        return 80


class FootprintAdditionalTwentyManager(FootprintSumManager):
    def types(self):
        return [
                'Proposición no de Ley ante el Pleno',
                'Proposición no de Ley en Comisión',
                ]

    def status(self):
        return 'Aprobada'

    def multiply(self):
        return 20


class FootprintAdditionalSixtyManager(FootprintSumManager):
    def types(self):
        return [
                'Proposición de ley de Grupos Parlamentarios del Congreso',
                'Proposición de ley de Diputados',
                ]

    def status(self):
        return 'Aprobada'

    def multiply(self):
        return 60


class FootprintInactivityPenalty():
    def __init__(self, topic, entity, typeof):
        self.topic = topic
        self.entity = entity
        self.typeof = typeof
        self.today = datetime.today()
        self.DAYS_IN_MONTH = 30

    def __months(self, months):
        return timedelta(days=self.DAYS_IN_MONTH*months)

    def more_than_twelve(self, date):
        return date <= (self.today - self.__months(12))

    def less_than_twelve(self, date):
        return date >= (self.today - self.__months(12))

    def less_than_six(self, date):
        return date >= (self.today - self.__months(6))

    def less_than_three(self, date):
        return date >= (self.today - self.__months(3))

    def compute(self):
        last_date = Initiatives.get_last_valid_creation_date(
                entity=self.entity,
                topic=self.topic,
                typeof=self.typeof)
        if not last_date:
            return 0
        if self.more_than_twelve(last_date) > 0 and self.less_than_twelve(last_date) == 0:
            return 0.50
        if self.less_than_twelve(last_date) > 0 and self.less_than_six(last_date) == 0:
            return 0.25
        if self.less_than_six(last_date) > 0 and self.less_than_three(last_date) == 0:
            return 0.10
        return 0


class FootprintDeputyManager:
    def __init__(self, deputy):
        self.deputy = deputy

    def __exists(self, field):
        return field in self.deputy

    def __not_empty(self, field):
        return self.deputy[field] != ''

    def __has_field(self, field):
        return self.__exists(field) and self.__not_empty(field)

    def has_email(self):
        return self.__has_field('email')

    def has_social(self):
        return self.__has_field('twitter') or self.__has_field('facebook')

    def compute_email(self):
        if self.has_email():
            return 40
        return 0

    def compute_social(self):
        if self.has_social():
            return 40
        return 0
