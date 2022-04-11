from typing import List

from Box2D import b2Body, b2Contact, b2ContactListener, b2Vec2

from boxdef import BodyType, EffectType


class ContactListener(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

        self.contacts: List[dict] = list()

        # list with contact number in case of an agent contact
        self.agent_contacts_num = list()
        # list with contact number in case of a border contact
        # agent contacts are not copied here
        self.border_contacts_num = list()

        # creating list with all possible contact types
        self.contact_types = list()  # list of lists [BodyType, BodyType]
        contact_number = 0
        type_list = list(BodyType)
        for type_a in list(BodyType):
            for type_b in type_list:
                contact_type = [type_a, type_b]
                self.contact_types.append(contact_type)
                if BodyType.AGENT in contact_type:
                    self.agent_contacts_num.append(contact_number)
                elif BodyType.BORDER in contact_type:
                    self.border_contacts_num.append(contact_number)
                contact_number += 1
            # since [type_0, type_1] is equal to [type_1, type_0]
            type_list.remove(type_a)

    def BeginContact(self, contact: b2Contact):
        # keeping track of which body is in touch with others
        self.add_contact(contact)
        self.handle_contact_begin(contact)
        pass

    def EndContact(self, contact):
        # updating which body is in touch with others
        self.remove_contact(contact)
        pass

    def PreSolve(self, contact, oldMainfold):
        pass

    def PostSolve(self, contact, impulse):
        pass

    def add_contact(self, contact: b2Contact):
        contact.fixtureA.body.userData.contacts.append(
            contact.fixtureB.body)
        contact.fixtureB.body.userData.contacts.append(
            contact.fixtureA.body)
        self.contacts.append(
            {"bodyA": contact.fixtureA.body, "bodyB": contact.fixtureB.body})

    def remove_contact(self, contact: b2Contact):
        for c in self.contacts:
            if c["bodyA"] == contact.fixtureA.body and c["bodyB"] == contact.fixtureB.body:
                contact.fixtureA.body.userData.contacts.remove(
                    contact.fixtureB.body)
                contact.fixtureB.body.userData.contacts.remove(
                    contact.fixtureA.body)
                self.contacts.remove(
                    {"bodyA": contact.fixtureA.body, "bodyB": contact.fixtureB.body})
                break

    def get_contact_number(self, contact):
        type_a = contact.fixtureA.body.userData.type
        type_b = contact.fixtureB.body.userData.type
        types = [type_a, type_b]

        for tix, contact_type in enumerate(self.contact_types):
            if set(contact_type) == set(types):
                return tix

    def effect(self, body: b2Body, effect_type: EffectType, new_value=None):
        # TODO: new value check
        if effect_type == EffectType.SET_VELOCITY:
            # new_value: b2Vec2
            body.linearVelocity = new_value
        if effect_type == EffectType.APPLY_FORCE:
            # new_value: [float, float]
            body.ApplyForce(
                force=b2Vec2(new_value), point=body.position, wake=True)
            print("Force applied: {}".format(new_value))
        elif effect_type == EffectType.DONE:
            self.env.done = True
        elif effect_type == EffectType.RESET:
            self.env.reset()
        pass

    def handle_contact_begin(self, contact: b2Contact):
        contact_num = self.get_contact_number(contact)

        if contact_num in self.agent_contacts_num:
            self.handle_agent_contact(contact)
        elif contact_num in self.border_contacts_num:
            self.handle_border_contact(contact)
        pass

    def handle_agent_contact(self, contact: b2Contact):

        if contact.fixtureA.body.userData.type == BodyType.AGENT:
            agent = contact.fixtureA.body
            body = contact.fixtureB.body
        else:
            body = contact.fixtureA.body
            agent = contact.fixtureB.body

        # print("effect: " + str(body.userData.effect) + str(agent.userData.effect))
        # get effect type
        effect_type = body.userData.effect["type"]

        # get value if needed
        value = body.userData.effect["value"]
        print("effect: " + str(EffectType(effect_type).name) + str(value))

        # perform effect
        self.effect(agent, effect_type, value)
        pass

    def handle_border_contact(self, contact):
        if contact.fixtureA.body.userData.type == BodyType.BORDER:
            # body B is the moving one
            body = contact.fixtureB.body
        else:
            # body A is the moving one
            body = contact.fixtureA.body

        value = contact.fixtureA.body.linearVelocity * -1
        self.effect(body, EffectType.SET_VELOCITY, value)
        pass
