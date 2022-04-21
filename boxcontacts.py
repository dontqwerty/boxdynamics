import logging
import math
from typing import List

from Box2D import b2Body, b2Contact, b2ContactListener, b2Vec2

from boxdef import BodyData, BodyType, EffectType, EffectWhen, EffectWho
from boxutils import anglemag_to_vec


class ContactListener(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env
        self.contacts: List[dict] = list()

    def BeginContact(self, contact: b2Contact):
        # keeping track of which body is in touch with others
        self.add_contact(contact)

        # performing bodyA effect
        self.contact_effect(contact.fixtureA.body,
                            contact.fixtureB.body, EffectWhen.ON_CONTACT)
        # performing bodyB effect
        self.contact_effect(contact.fixtureB.body,
                            contact.fixtureA.body, EffectWhen.ON_CONTACT)

        pass

    def EndContact(self, contact):
        # updating which body is in touch with others
        self.remove_contact(contact)

        # performing bodyA effect
        self.contact_effect(contact.fixtureA.body,
                            contact.fixtureB.body, EffectWhen.OFF_CONTACT)
        # performing bodyB effect
        self.contact_effect(contact.fixtureB.body,
                            contact.fixtureA.body, EffectWhen.OFF_CONTACT)
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

    def contact_effect(self, bodyA: b2Body, bodyB: b2Body, when: EffectWhen):
        dataA: BodyData = bodyA.userData
        dataB: BodyData = bodyB.userData

        effect_typeA = dataA.effect["type"]
        effect_whoA = dataA.effect["who"]
        effect_whenA = dataA.effect["when"]

        # checking effect when
        if effect_whenA == when:
            # checking effect who
            if effect_whoA == EffectWho.BOTH or \
                (effect_whoA == EffectWho.AGENT and dataB.type == BodyType.AGENT) or \
                    (effect_whoA == EffectWho.OTHER and dataB.type != BodyType.AGENT):
                param_0 = dataA.effect["param_0"]
                param_1 = dataA.effect["param_1"]
                # performing fixtureA effect
                logging.debug("perfmorming effect {} on {} at {}".format(effect_typeA.name, effect_whoA.name, effect_whenA.name))
                if effect_typeA == EffectType.NONE:
                    pass
                elif effect_typeA == EffectType.APPLY_FORCE:
                    # calculating force based on angle (param_0)
                    # and mag (param_1)
                    force = anglemag_to_vec(angle=param_1, magnitude=param_1)
                    bodyB.ApplyForce(
                        force=force, point=bodyB.position, wake=True)
                elif effect_typeA == EffectType.SET_VELOCITY:
                    vel = anglemag_to_vec(angle=param_0, magnitude=param_1)
                    bodyB.linearVelocity = vel
                    pass
                elif effect_typeA == EffectType.INVERT_VELOCITY:
                    bodyB.linearVelocity = bodyB.linearVelocity * (-1)
                    pass
                elif effect_typeA == EffectType.BOUNCE:
                    # TODO: bounce
                    pass
                elif effect_typeA == EffectType.SET_LIN_DAMP:
                    bodyB.linearDamping = param_0
                    pass
                elif effect_typeA == EffectType.SET_ANG_DAMP:
                    bodyB.angularDamping = param_0
                    pass
                elif effect_typeA == EffectType.SET_MAX_ACTION:
                    # TODO: max action
                    pass
                elif effect_typeA == EffectType.SET_FRICTION:
                    bodyB.fixtures[0].friction = param_0
                    pass
                elif effect_typeA == EffectType.DONE:
                    self.env.done = True
                    pass
                elif effect_typeA == EffectType.RESET:
                    self.env.reset()
                    pass
                else:
                    assert False and "EffectType not supported"
        pass
